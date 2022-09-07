import torch.nn as nn


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

        