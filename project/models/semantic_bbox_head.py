import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from utils import ConvModule
import numpy as np
from loss.mse_loss import MSELoss
from loss.cross_entropy_loss import CrossEntropyLoss
from loss.smooth_l1_loss import SmoothL1Loss

class SemanticBBoxHead(nn.Module):
    def __init__(self, 
        with_avg_pool=False,
        with_reg=True,
        with_semantic=True,
        roi_feat_size=14,
        in_channels=2048,
        num_classes=4, # keycap dataset
        semantic_dims=300,
        seen_class=True,
        gzsd=False,
        reg_with_semantic=False,
        share_semantic=False,
        voc_path='vb/vocabulary_w2v.txt',
        vec_path='vb/word_w2v_withbg_65_15.txt',
        use_lsoftmax=False,
        with_decoder=True,
        sync_bg=True, # reference from zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_65_15_decoder_notanh.py
        semantic_norm=False,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        num_shared_convs=0,
        num_shared_fcs=2, # reference from zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_65_15_decoder_notanh.py
        num_semantic_convs=0,
        num_semantic_fcs=0,
        num_reg_convs=0,
        num_reg_fcs=0,
        conv_out_channels=256,
        fc_out_channels=1024,
        conv_cfg=None,
        norm_cfg=None,
        loss_bbox=SmoothL1Loss(beta=1.0, loss_weight=1.0),
        loss_semantic=CrossEntropyLoss(use_mask=False, loss_weight=1.0),
        loss_ed=MSELoss(reduction='mean', loss_weight=0.5)
    ):
        super(SemanticBBoxHead, self).__init__()
       
        self.seen_class = seen_class
        self.gzsd = gzsd
        self.reg_with_semantic = reg_with_semantic
        self.share_semantic = share_semantic
        self.with_avg_pool = with_avg_pool
        self.with_reg = with_reg
        self.with_semantic = with_semantic
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False
        self.use_lsoftmax = use_lsoftmax
        self.with_decoder = with_decoder
        self.semantic_norm = semantic_norm
        self.sync_bg = sync_bg
        self.loss_bbox = loss_bbox
        self.loss_semantic = loss_semantic
        self.loss_ed = loss_ed
        assert with_reg or with_semantic
        assert (num_shared_convs + num_shared_fcs + num_semantic_convs +
                num_semantic_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_semantic_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_semantic:
            assert num_semantic_convs == 0 and num_semantic_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
            

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_semantic_convs = num_semantic_convs
        self.num_semantic_fcs = num_semantic_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # self.loss_bbox = build_loss(loss_bbox)
        # self.loss_semantic = build_loss(loss_semantic)
        # self.loss_ed = build_loss(loss_ed)

        in_channels = self.in_channels
        in_channels *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)

        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)

        if self.with_semantic:
            self.fc_semantic = nn.Linear(self.in_channels, semantic_dims)
            # voc = np.loadtxt('MSCOCO/vocabulary_w2v.txt', dtype='float32', delimiter=',')
            if voc_path is not None:
                voc = np.loadtxt(voc_path, dtype='float32', delimiter=',')
            else:
                voc = None
            # vec = np.loadtxt('MSCOCO/word_w2v.txt', dtype='float32', delimiter=',')
            vec_load = np.loadtxt(vec_path, dtype='float32', delimiter=',')
            # if self.seen_class:
            vec = vec_load[:, :num_classes]
            # else:
            vec_unseen = np.concatenate([vec_load[:, 0:1], vec_load[:, num_classes:]], axis=1)
            vec = torch.tensor(vec, dtype=torch.float32)
            if voc is not None:
                voc = torch.tensor(voc, dtype=torch.float32)
            vec_unseen = torch.tensor(vec_unseen, dtype=torch.float32)
            self.vec = vec.cuda()  # 300*n
            if voc is not None:
                self.voc = voc.cuda()  # 300*66
            else:
               self.voc = None
            self.vec_unseen = vec_unseen.cuda()

            if self.voc is not None:
                self.kernel_semantic = nn.Linear(self.voc.shape[1], self.vec.shape[0]) #n*300
        
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add semantic specific branch
        self.semantic_convs, self.semantic_fcs, self.semantic_last_dim = \
            self._add_conv_fc_branch(
                self.num_semantic_convs, self.num_semantic_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)


        

        # reconstruct fc_semantic and fc_reg since input channels are changed
        if self.with_semantic:
            self.fc_semantic = nn.Linear(self.semantic_last_dim, semantic_dims)
            if self.with_decoder:
                self.d_fc_semantic = nn.Linear(semantic_dims, self.semantic_last_dim)
            if self.voc is not None:
                self.kernel_semantic = nn.Linear(self.voc.shape[1], self.vec.shape[0])  # n*300
                if self.with_decoder:
                    self.d_kernel_semantic = nn.Linear(self.vec.shape[0], self.voc.shape[1])  # n*300
            else:
                self.kernel_semantic = nn.Linear(self.vec.shape[1], self.vec.shape[1])
                if self.with_decoder:
                    self.d_kernel_semantic = nn.Linear(self.vec.shape[1], self.vec.shape[1])  # n*300

        if self.with_reg and not self.reg_with_semantic:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        self.fc_res = nn.Linear(self.vec.shape[0], self.vec.shape[0])
        # self.fc_res = nn.Linear(self.semantic_last_dim, self.vec.shape[0])

        
    def forward(self, x, bg_vector=None):

        if self.num_shared_fcs > 0:
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches for "regression branch and semantic transform"
        x_semantic = x
        x_reg = x

        # visual feature to regression branch
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # visual feature to semantic feature ( x_semantic -> semantic_feature)
        for conv in self.semantic_convs:
            x_semantic = conv(x_semantic)
        if x_semantic.dim() > 2:
            x_semantic = x_semantic.view(x_semantic.size(0), -1)
        for fc in self.semantic_fcs:
            x_semantic = self.relu(fc(x_semantic))

        if self.with_semantic:
            semantic_feature = self.fc_semantic(x_semantic)
            if self.sync_bg:
                with torch.no_grad():
                    xxxx = self.vec[:, 0]
                    self.vec[:, 0] = bg_vector
                    if not self.seen_class:
                        self.vec_unseen[:, 0] = bg_vector
        
        # predict semantic score
        if self.voc is not None:
            # matrix multiplication (semantic_feature, word-vector)
            semantic_score = torch.mm(semantic_feature, self.voc)
            semantic_score = self.kernel_semantic(semantic_score)

            # decode
            if self.with_decoder:
                d_semantic_score = self.d_kernel_semantic(semantic_score)
                d_semantic_feature = torch.mm(d_semantic_score, self.voc.t())
                d_semantic_feature = self.d_fc_semantic(d_semantic_feature)
        
        # predict bbox
        bbox_pred = self.fc_reg(x_reg)

        if self.with_decoder:
            return semantic_score, bbox_pred, x_semantic, d_semantic_feature
        else:
            return semantic_score, bbox_pred
    
    def compute_reconstructed_error(self, x, d_x):
        self.reconstructed_error = self.loss_ed(x, d_x)
        return self.reconstructed_error
    
    def compute_semantic_loss(self, semantic_score, bbox_pred, bbox_target):
        self.semantic_loss = self.loss_semantic(pred, label, weight=None, reduction='mean', avg_factor=None)

    
    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

if __name__ == '__main__':
    ZSD = SemanticBBoxHead().cuda()
    x = torch.randn(1, 2048, 14, 14).cuda()
    bg = torch.randn(300).cuda()
    y = ZSD(x, bg)
    
        

        