from typing import Optional, Union
import torch
import torch.nn as nn

from project.data.image import ImageAnnotation, ImageMetadata

from .utils.anchor import AnchorGenerator

from .backbone import Extractor
from .ba_rpn import BackgroundAwareRPNHead, BackgroundAwareRPN


class ZeroShotMaskRCNN(nn.Module):
    def __init__(
        self,
        vec_path: str,
        voc_path: Optional[str] = None,
        backbone_name: str = "resnet50",
        backbone_trainable_layers: int = 3,
        backbone_returned_layers: Optional[list[int]] = None,
        # Anchor generator
        anchor_sizes: Union[list[int], tuple[int, ...]] = (4, 8, 16, 32, 64),
        anchor_scales: Union[list[int], tuple[int, ...]] = (8, 16, 32),
        anchor_ratios: Union[list[float], tuple[float, ...]] = (0.5, 1.0, 2.0),
        # RPN Training
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        rpn_num_sampled_anchors: int = 256,
        rpn_sampled_positive_fraction: float = 0.5,
        rpn_train_pre_nms_top_n: int = 2000,
        rpn_train_post_nms_top_n: int = 2000,
        # RPN Inference
        rpn_test_pre_nms_top_n: int = 1000,
        rpn_test_post_nms_top_n: int = 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_cls_score_thresh: float = 0.0,
        rpn_min_bbox_size: float = 0.0,
        **kwargs
    ) -> None:
        self.backbone = Extractor(
            backbone_name=backbone_name,
            trainable_layers=backbone_trainable_layers,
            returned_layers=backbone_returned_layers,
        )
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, scales=anchor_scales, ratios=anchor_ratios
        )
        rpn_head = BackgroundAwareRPNHead(
            in_channels=self.backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_size,
            vec_path=vec_path,
            voc_path=voc_path,
        )
        self.rpn = BackgroundAwareRPN(
            head=rpn_head,
            anchor_generator=anchor_generator,
            fg_iou_thresh=rpn_fg_iou_thresh,
            bg_iou_thresh=rpn_bg_iou_thresh,
            batch_size_per_image=rpn_num_sampled_anchors,
            positive_fraction=rpn_sampled_positive_fraction,
            train_pre_nms_top_n=rpn_train_pre_nms_top_n,
            train_post_nms_top_n=rpn_train_post_nms_top_n,
            test_pre_nms_top_n=rpn_test_pre_nms_top_n,
            test_post_nms_top_n=rpn_test_post_nms_top_n,
            nms_thresh=rpn_nms_thresh,
            cls_score_thresh=rpn_cls_score_thresh,
            min_bbox_size=rpn_min_bbox_size,
        )

    def forward(self, imgs: torch.Tensor, img_metas: list[ImageMetadata], targets: list[ImageAnnotation]):
        feature_maps = self.backbone(imgs)
        proposals, bg_vecs, rpn_losses = self.rpn(feature_maps, imgs, img_metas, targets)
