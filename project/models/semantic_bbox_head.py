import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


class SemanticBBoxHead(nn.Module):
    def __init__(self, 
        with_avg_pool=False,
        with_reg=True,
        with_semantic=True,
        roi_feat_size=7,
        in_channels=256,
        num_classes=66,
        semantic_dims=300,
        seen_class=True,
        gzsd=False,
        reg_with_semantic=False,
        share_semantic=False,
        voc_path=None,
        vec_path=None,
        use_lsoftmax=False,
        with_decoder=False,
        sync_bg=False,
        semantic_norm=False,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1.0
        ),
        loss_semantic=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        ),
        loss_ed=dict(type='MSELoss', loss_weight=0.5),
    ):
        super(SemanticBBoxHead, self).__init__()
        assert with_reg or with_semantic
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

        # self.loss_bbox = build_loss(loss_bbox)
        # self.loss_semantic = build_loss(loss_semantic)
        # self.loss_ed = build_loss(loss_ed)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

        
    def forward(self, x):

        # separate branches for "regression branch and semantic transform"
        x_semantic = x
        x_reg = x

        # visual feature to regression branch
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # visual feature to semantic feature ( x_semantic -> semantic_feature)
        for conv in self.semantic_convs:
            x_semantic = conv(x_semantic)
        if x_semantic.dim() > 2:
            if self.with_avg_pool:
                x_semantic = self.avg_pool(x_semantic)
            x_semantic = x_semantic.view(x_semantic.size(0), -1)
        for fc in self.semantic_fcs:
            x_semantic = self.relu(fc(x_semantic))

        if self.with_semantic:
            semantic_feature = self.fc_semantic(x_semantic)
            if self.sync_bg:
                with torch.no_grad():
                    self.vec[:, 0] = bg_vector
                    if not self.seen_class:
                        self.vec_unseen[:, 0] = bg_vector
        
        if self.voc is not None:
            # matrix multiplication (semantic_feature, word-vector)
            semantic_score = torch.mm(semantic_feature, self.voc)

        

        