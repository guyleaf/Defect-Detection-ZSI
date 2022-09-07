from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops
from torchvision.ops import Conv2dNormActivation

from ..data import ImageAnnotation, ImageMetadata
from .utils import (
    AnchorGenerator,
    BoxCoder,
    Matcher,
    BalancedPositiveNegativeSampler,
    concat_box_prediction_layers,
    load_word_vectors,
    multi_apply,
)


class BackgroundAwareRPNHead(nn.Module):
    _voc: Optional[torch.Tensor]

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        vec_path: str,
        voc_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        # in source code: padding = 1
        self.conv = Conv2dNormActivation(
            in_channels, in_channels, 3, norm_layer=None
        )

        vec, voc = self._load_word_vectors(vec_path, voc_path)
        bg_vec = vec[0]

        semantic_channels = vec.size(1)
        # visual feature -> semantic_encoder -> semantice feature -> cls_logits
        self.semantic_encoder = nn.Conv2d(in_channels, semantic_channels, 1)
        self.cls_logits = nn.Conv2d(semantic_channels, num_anchors * 2, 1)

        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

        if voc is not None:
            hidden_channels = voc.size(0)
            self.voc_convs = nn.Sequential(
                nn.Conv2d(semantic_channels, hidden_channels, 1, bias=False),
                nn.Conv2d(hidden_channels, semantic_channels, 1)
            )
        else:
            self.voc_convs = None

        self.register_buffer("_voc", voc, persistent=False)

        # initialize weights
        self.apply(self._init_weights)
        if self.voc_convs is not None:
            self._init_voc_convs(voc)
        self._init_cls_logits(bg_vec)

    @torch.no_grad()
    def _init_voc_convs(self, voc: torch.Tensor):
        weights = self.voc_convs[0]
        weights.data = voc.unsqueeze(-1).unsqueeze(-1)
        for param in self.voc_convs[0].parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _init_cls_logits(self, bg_vec: torch.Tensor):
        weights = self.cls_logits.weight
        weights.data[::2] = bg_vec.tile((weights.size(0) // 2, 1)).unsqueeze(-1).unsqueeze(-1)

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=0.01)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _load_word_vectors(
        self, vec_path: str, voc_path: Optional[str]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        vec = load_word_vectors(vec_path, as_torch=True)
        voc = None
        if voc_path:
            voc = load_word_vectors(voc_path, as_torch=True)

        return vec, voc

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x)

        # classification branch
        semantic_feature = self.semantic_encoder(x)
        if self.voc_convs is not None:
            semantic_feature = self.voc_convs(semantic_feature)
        cls_score = self.cls_logits(semantic_feature)

        # regression branch
        bbox_pred = self.bbox_pred(x)

        bg_vec = torch.mean(self.cls_logits.weight.data[::2], dim=0)
        return cls_score, bbox_pred, bg_vec


class BackgroundAwareRPN(nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Args:
        head (nn.Module): module that computes the objectness and regression deltas
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
    """

    def __init__(
        self,
        head: nn.Module,
        anchor_generator: AnchorGenerator,
        # Faster-RCNN Training
        fg_iou_thresh: float = 0.7,
        bg_iou_thresh: float = 0.3,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        train_pre_nms_top_n: int = 2000,
        train_post_nms_top_n: int = 2000,
        # Faster-RCNN Inference
        test_pre_nms_top_n: int = 1000,
        test_post_nms_top_n: int = 1000,
        nms_thresh: float = 0.7,
        score_thresh: float = 0.0,
        min_size: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = head
        self.anchor_generator = anchor_generator
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.box_similarity = box_ops.box_iou

        # TODO: implement min_pos_iou
        # In ZSI paper, they implement min_pos_iou to assign low quality matches to gt
        # when positive samples can have smaller IoU than pos_iou_thr
        self.proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        self._train_pre_nms_top_n = train_pre_nms_top_n
        self._train_post_nms_top_n = train_post_nms_top_n
        self._test_pre_nms_top_n = test_pre_nms_top_n
        self._test_post_nms_top_n = test_post_nms_top_n

        self._nms_thresh = nms_thresh
        self._score_thresh = score_thresh
        self._min_size = min_size

    @property
    def pre_nms_top_n(self):
        if self.training:
            return self._train_pre_nms_top_n
        return self._test_pre_nms_top_n

    @property
    def post_nms_top_n(self):
        if self.training:
            return self._train_post_nms_top_n
        return self._test_post_nms_top_n

    def assign_targets_to_anchors(
        self, anchors: list[torch.Tensor], targets: list[ImageAnnotation]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image.bboxes

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(
                    anchors_per_image.shape, dtype=torch.float32, device=device
                )
                labels_per_image = torch.zeros(
                    (anchors_per_image.shape[0],),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                match_quality_matrix = self.box_similarity(
                    gt_boxes, anchors_per_image
                )
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -1, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[
                    matched_idxs.clamp(min=0)
                ]

                labels_per_image = matched_idxs.to(dtype=torch.float32)

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(
        self, cls_scores: torch.Tensor, num_anchors_per_level: list[int]
    ) -> torch.Tensor:
        r = []
        offset = 0
        for ob in cls_scores.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(
        self,
        proposals: torch.Tensor,
        cls_scores: torch.Tensor,
        image_metas: list[ImageMetadata],
        num_anchors_per_level: list[int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        num_images = proposals.shape[0]
        device = proposals.device

        # only use the foreground scores (objectness probability)
        objectness_prob = cls_scores.softmax(dim=1)[:, 1]
        objectness_prob = objectness_prob.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness_prob)
        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness_prob, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness_prob = objectness_prob[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, image_meta in zip(
            proposals, objectness_prob, levels, image_metas
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_meta.size)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self._min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # Note: it is the default implementation of Faster R-CNN
            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self._score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self._nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n]

            # TODO: implement threshold filters when nms_across_levels = True/False
            # current implementation is for nms_across_levels = True
            # keep = keep[: self._max_num]
            # boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(
        self,
        cls_scores: torch.Tensor,
        pred_bbox_deltas: torch.Tensor,
        labels: list[torch.Tensor],
        regression_targets: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cls_scores (Tensor)
            pred_bbox_deltas (Tensor)
            labels (list[Tensor])
            regression_targets (list[Tensor])
        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds.numel())

        objectness_loss = F.cross_entropy(
            cls_scores[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        images: torch.Tensor,
        image_metas: list[ImageMetadata],
        targets: Optional[list[ImageAnnotation]] = None,
    ) -> tuple:
        cls_scores, bbox_preds, bg_vecs = multi_apply(self.head, feature_maps)
        anchors: list[torch.Tensor] = self.anchor_generator(
            feature_maps, images
        )

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in bbox_preds]
        # s[0] = 4 * num_anchors
        num_anchors_per_level = [
            (s[0] // 4) * s[1] * s[2]
            for s in num_anchors_per_level_shape_tensors
        ]
        cls_scores, bbox_preds = concat_box_prediction_layers(
            cls_scores, bbox_preds
        )

        # apply bbox_preds to anchors to obtain the decoded proposals
        # note that we detach the preds because Faster R-CNN do not backprop through
        # the proposals
        # bbox_preds: (center-x, center-y, w, h)
        # anchors: (x1, y1, x2, y2)
        # (center-x, center-y, w, h) -> (x1, y1, x2, y2)
        proposals = self.box_coder.decode(bbox_preds.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # do not backprop through cls_scores
        boxes, _ = self.filter_proposals(
            proposals,
            cls_scores.detach(),
            image_metas,
            num_anchors_per_level,
        )

        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            labels, matched_gt_boxes = self.assign_targets_to_anchors(
                anchors, targets
            )
            regression_targets = self.box_coder.encode(
                matched_gt_boxes, anchors
            )
            loss_cls, loss_bbox = self.compute_loss(
                cls_scores, bbox_preds, labels, regression_targets
            )
            losses = {
                "loss_rpn_cls": loss_cls,
                "loss_rpn_bbox": loss_bbox,
            }
        return boxes, bg_vecs, losses
