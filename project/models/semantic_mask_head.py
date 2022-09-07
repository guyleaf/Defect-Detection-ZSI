from doctest import testfile
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import numpy as np
import mmcv

from utils import ConvModule
from utils import mask_target
from loss.mse_loss import MSELoss
from loss.cross_entropy_loss import CrossEntropyLoss


class SemanticMaskHead(nn.Module):
    def __init__(self, 
        num_convs=4,
        roi_feat_size=14,
        in_channels=2048,
        conv_kernel_size=3,
        conv_out_channels=2048,
        upsample_method='deconv',
        upsample_ratio=2,
        num_classes=4, # modify from 81 -> 4
        semantic_dims=300,
        seen_class=True,
        gzsd=False,
        share_semantic=False,
        sync_bg=True, # reference from zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_65_15_decoder_notanh.py
        voc_path=None,
        vec_path='../data/word_w2v_withbg_65_15.txt', # reference from zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_65_15_decoder_notanh.py
        with_learnable_kernel=True,
        with_decoder=True, # reference from zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_65_15_decoder_notanh.py
        class_agnostic=False,
        conv_cfg=None,
        norm_cfg=None,
        loss_mask=CrossEntropyLoss(use_mask=True, loss_weight=1.0),
        loss_ed=MSELoss(reduction='mean', loss_weight=0.5)
    ):
        super(SemanticMaskHead, self).__init__()
        self.seen_class = seen_class
        self.gzsd = gzsd
        self.share_semantic = share_semantic
        self.with_learnable_kernel = with_learnable_kernel
        self.with_decoder = with_decoder
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = loss_mask
        self.loss_ed = loss_ed
        self.sync_bg=sync_bg
        

        # conv for upsampling
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg
                )
            )
        # convT for encoder
        self.convT = ConvModule(
                in_channels,
                300,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg
            )
        # dconvT for decoder
        if self.with_decoder:
            self.dconvT = ConvModule(
                        300,
                        in_channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg
                    )

        upsample_in_channels = (self.conv_out_channels if self.num_convs > 0 else in_channels)

        if self.upsample_method is None:
            self.upsample = None

        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                    upsample_in_channels,
                    self.conv_out_channels,
                    self.upsample_ratio,
                    stride=self.upsample_ratio
                )
        else:
            self.upsample = nn.Upsample(scale_factor=self.upsample_ratio, mode=self.upsample_method)

        # relu
        self.relu = nn.ReLU(inplace=True)

        # word related
        if voc_path is not None:
            voc = np.loadtxt(voc_path, dtype='float32', delimiter=',')
        else:
            voc = None

        vec_load = np.loadtxt(vec_path, dtype='float32', delimiter=',')

        vec = vec_load[:, :num_classes]
        vec_unseen = np.concatenate([vec_load[:, 0:1], vec_load[:, num_classes:]], axis=1)
        vec = torch.tensor(vec, dtype=torch.float32)
        if voc is not None:
            voc = torch.tensor(voc, dtype=torch.float32)
        vec_unseen = torch.tensor(vec_unseen, dtype=torch.float32)
        self.vec_unseen = vec_unseen.cuda()
        self.vec = vec.cuda()  # 300*n
        self.conv_vec = nn.Conv2d(300, num_classes, 1, bias=False)

        self.conv_vec.weight.data = torch.unsqueeze(torch.unsqueeze(self.vec.t(), -1), -1)

        if not self.seen_class:
            self.con_vec_t = nn.Conv2d(num_classes, 300, 1, bias=False)
            self.con_vec_t.weight.data = torch.unsqueeze(torch.unsqueeze(self.vec, -1), -1)
            self.conv_vec_unseen = nn.Conv2d(300, vec_unseen.shape[1], 1, bias=False)
            self.conv_vec_unseen.weight.data = torch.unsqueeze(torch.unsqueeze(self.vec_unseen.t(), -1), -1)
        
        if voc is not None:
            self.voc = voc.cuda()  # 300*66
            self.conv_voc = nn.Conv2d(300, self.voc.size(1), 1, bias=False)
            self.conv_voc.weight.data = torch.unsqueeze(torch.unsqueeze(self.voc.t(), -1), -1)
        else:
            self.voc = None

        self.vec_unseen = vec_unseen.cuda()
        if self.with_learnable_kernel:
            if self.voc is not None:
                self.kernel_semantic = nn.Conv2d(self.voc.size(1), 300, kernel_size=3, padding=1)
                if self.with_decoder:
                    self.d_kernel_semantic = nn.Conv2d(300, self.voc.size(1), kernel_size=3, padding=1)
            else:
                self.kernel_semantic = nn.Conv2d(300, 300, kernel_size=3, padding=1)
                if self.with_decoder:
                    self.d_kernel_semantic = nn.Conv2d(300, 300, kernel_size=3, padding=1)


    def forward(self, x, y, bg_vector=None):
        '''
        input x : visual feature (batch, 2048, 14, 14)
        input y : target mask (batch, 28, 28)
        input bg_vector : (300)
        '''
        # replace bg by ba-rpn
        if self.sync_bg:
            with torch.no_grad():
                self.conv_vec.weight.data[0] = bg_vector[0]
                if not self.seen_class:
                    self.conv_vec_unseen.weight.data[0] = bg_vector[0]
        
        # upsampling 
        for conv in self.convs:
            conv4_x = conv(x)
        if self.upsample is not None:
            conv4_x = self.upsample(conv4_x)
            if self.upsample_method == 'deconv':
                conv4_x = self.relu(conv4_x)

        # encoder
        x = self.convT(conv4_x)
        if self.voc is not None:
            x = self.conv_voc(x)
        if self.with_learnable_kernel:
            x = self.kernel_semantic(x)

        # decoder
        if self.with_decoder:
            if self.with_learnable_kernel:
                d_x = self.d_kernel_semantic(x)
            d_x = self.dconvT(d_x)

        # classification module
        mask_pred_seen = self.conv_vec(x)
        if not self.seen_class and not self.gzsd:
            mask_pred = self.con_vec_t(mask_pred_seen)
            mask_pred_unseen = self.conv_vec_unseen(mask_pred)
            return mask_pred_unseen
        if self.gzsd:
            mask_pred = self.con_vec_t(mask_pred_seen)
            mask_pred_unseen = self.conv_vec_unseen(mask_pred)
            return mask_pred_seen, mask_pred_unseen
        
        # compute loss
        loss = {
            "reconstructed_error" : self.compute_reconstructed_error(conv4_x, d_x),
            "BCE_loss" :  self.compute_binary_cls_error(mask_pred_seen, y)
        }
        return mask_pred_seen, loss
        # if not self.with_decoder:
        #     return mask_pred_seen
        # else:
        #     return mask_pred_seen, conv4_x, d_x

    

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets
 
    def compute_reconstructed_error(self, x, d_x):
        self.reconstructed_error = self.loss_ed(x, d_x)
        return self.reconstructed_error
    
    def compute_binary_cls_error(self, x, y):
        '''
        input x : pred mask (batch, 28, 28)
        input y : target mask (batch, 28, 28)
        '''
        self.cls_loss = self.loss_mask(x, y)
        return self.cls_loss

    
    def init_weights(self):
        for m in [self.upsample]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        if self.with_learnable_kernel:
            for m in [self.kernel_semantic]:
                if m is None:
                    continue
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        for m in [self.conv_vec]:
            for param in m.parameters():
                param.requires_grad = False
        if self.voc is not None:
            for m in [self.conv_voc]:
                for param in m.parameters():
                    param.requires_grad = False

if __name__ == '__main__':
    SMH = SemanticMaskHead().cuda()
    x = torch.randn(1, 2048, 14, 14).cuda()
    y = torch.randn(1, 4, 28, 28).cuda()
    bg = torch.randn(300).cuda()
    xx = SMH(x, y, bg)
    
    x=0