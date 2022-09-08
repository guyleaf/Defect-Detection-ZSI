from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from .loss.cross_entropy_loss import CrossEntropyLoss
from .loss.mse_loss import MSELoss
from .loss.smooth_l1_loss import SmoothL1Loss
from .utils import ConvModule, bbox_target, load_word_vectors


class SemanticBBoxHead(nn.Module):
    vec: torch.Tensor
    vec_unseen: torch.Tensor
    voc: Optional[torch.Tensor]

    def __init__(
        self,
        vec_path: str,
        voc_path: Optional[str] = None,
        with_avg_pool: bool = False,
        roi_feat_size: int = 14,
        in_channels: int = 2048,
        num_classes: int = 4,  # keycap dataset
        semantic_dims: int = 300,
        seen_class: bool = True,
        share_semantic: bool = False,
        with_decoder: bool = True,
        sync_bg: bool = True,  # reference from zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_65_15_decoder_notanh.py
        target_means: tuple[float, float, float, float] = [0.0, 0.0, 0.0, 0.0],
        target_stds: tuple[float, float, float, float] = [0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic: bool = False,
        num_shared_convs: int = 0,
        num_shared_fcs: int = 2,  # reference from zero-shot-mask-rcnn-BARPN-bbox_mask_sync_bg_65_15_decoder_notanh.py
        num_semantic_convs: int = 0,
        num_semantic_fcs: int = 0,
        num_reg_convs: int = 0,
        num_reg_fcs: int = 0,
        conv_out_channels: int = 256,
        fc_out_channels: int = 1024,
        loss_bbox_weight: float = 1.0,
        loss_semantic_weight: float = 1.0,
        loss_ed_weight: float = 0.5,
    ):
        super(SemanticBBoxHead, self).__init__()

        self.seen_class = seen_class
        self.share_semantic = share_semantic
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.with_decoder = with_decoder
        self.sync_bg = sync_bg

        total = num_shared_convs
        total += num_shared_fcs
        total += num_semantic_convs
        total += num_semantic_fcs
        total += num_reg_convs
        total += num_reg_fcs
        assert total > 0

        if num_semantic_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_semantic_convs = num_semantic_convs
        self.num_semantic_fcs = num_semantic_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels

        self.loss_bbox = SmoothL1Loss(beta=1.0, loss_weight=loss_bbox_weight)
        self.loss_semantic = CrossEntropyLoss(
            use_sigmoid=False, loss_weight=loss_semantic_weight
        )
        self.loss_ed = MSELoss(reduction="mean", loss_weight=loss_ed_weight)

        in_channels = self.in_channels
        in_channels *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)

        out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
        self.fc_reg = nn.Linear(in_channels, out_dim_reg)

        self.fc_semantic = nn.Linear(self.in_channels, semantic_dims)
        if voc_path is not None:
            voc: torch.Tensor = (
                load_word_vectors(voc_path, as_torch=True).float().T
            )
        else:
            voc = None

        vec_load: torch.Tensor = load_word_vectors(
            vec_path, as_torch=True
        ).float()
        # if self.seen_class:
        vec = vec_load[:num_classes].T
        # else:
        vec_unseen = torch.cat(
            [vec_load[0, None], vec_load[num_classes:]], dim=0
        ).T

        if voc is not None:
            self.kernel_semantic = nn.Linear(
                voc.shape[1], voc.shape[0]
            )  # n*300

        # add shared convs and fcs
        (
            self.shared_convs,
            self.shared_fcs,
            last_layer_dim,
        ) = self._add_conv_fc_branch(
            self.num_shared_convs, self.num_shared_fcs, self.in_channels, True
        )
        self.shared_out_channels = last_layer_dim

        # add semantic specific branch
        (
            self.semantic_convs,
            self.semantic_fcs,
            self.semantic_last_dim,
        ) = self._add_conv_fc_branch(
            self.num_semantic_convs,
            self.num_semantic_fcs,
            self.shared_out_channels,
        )

        # add reg specific branch
        (
            self.reg_convs,
            self.reg_fcs,
            self.reg_last_dim,
        ) = self._add_conv_fc_branch(
            self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels
        )

        # reconstruct fc_semantic and fc_reg since input channels are changed
        self.fc_semantic = nn.Linear(self.semantic_last_dim, semantic_dims)
        if self.with_decoder:
            self.d_fc_semantic = nn.Linear(
                semantic_dims, self.semantic_last_dim
            )
        if voc is not None:
            self.kernel_semantic = nn.Linear(
                voc.shape[1], vec.shape[0]
            )  # n*300
            if self.with_decoder:
                self.d_kernel_semantic = nn.Linear(
                    vec.shape[0], voc.shape[1]
                )  # n*300
        else:
            self.kernel_semantic = nn.Linear(vec.shape[1], vec.shape[1])
            if self.with_decoder:
                self.d_kernel_semantic = nn.Linear(
                    vec.shape[1], vec.shape[1]
                )  # n*300

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        self.fc_res = nn.Linear(vec.shape[0], vec.shape[0])
        # self.fc_res = nn.Linear(self.semantic_last_dim, self.vec.shape[0])

        self.register_buffer("vec", vec)
        self.register_buffer("vec_unseen", vec_unseen)
        self.register_buffer("voc", voc)

    def forward(
        self,
        x: torch.Tensor,
        bg_vector: torch.Tensor,
        targets: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
    ):
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

        semantic_feature = self.fc_semantic(x_semantic)
        if self.sync_bg:
            with torch.no_grad():
                self.vec[:, 0] = bg_vector
                if not self.seen_class:
                    self.vec_unseen[:, 0] = bg_vector

        # predict semantic score
        if self.voc is not None:
            # matrix multiplication (semantic_feature, word-vector)
            semantic_score = torch.mm(semantic_feature, self.voc)
            semantic_score = self.kernel_semantic(semantic_score)

            # decode
            d_semantic_feature = None
            if self.training and self.with_decoder:
                d_semantic_score = self.d_kernel_semantic(semantic_score)
                d_semantic_feature = torch.mm(d_semantic_score, self.voc.t())
                d_semantic_feature = self.d_fc_semantic(d_semantic_feature)

            semantic_score = torch.mm(semantic_score, self.vec)
        else:
            semantic_score = self.kernel_semantic(self.vec)
            semantic_score = torch.tanh(semantic_score)
            semantic_score = torch.mm(semantic_feature, semantic_score)

        # predict bbox
        bbox_pred = self.fc_reg(x_reg)

        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")

            labels, label_weights, bbox_targets, bbox_weights = targets
            losses = {
                "zsd_semantic_loss": self.compute_semantic_loss(
                    semantic_score, labels, label_weights
                ),
                "zsd_bbox_loss": self.compute_bbox_loss(
                    bbox_pred, labels, bbox_targets, bbox_weights
                ),
            }

            if self.with_decoder:
                reconstructed_error = self.compute_reconstructed_error(
                    x_semantic, d_semantic_feature
                )
                losses["zsd_reconstructed_error"] = reconstructed_error
        return semantic_score, bbox_pred, losses

    def get_target(
        self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg
    ):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        semantic_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds,
        )
        return semantic_reg_targets

    def compute_reconstructed_error(self, x: torch.Tensor, d_x: torch.Tensor):
        return self.loss_ed(x, d_x)

    def compute_semantic_loss(
        self,
        semantic_score: torch.Tensor,
        labels: torch.Tensor,
        label_weights: torch.Tensor,
    ):
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
        return self.loss_semantic(
            semantic_score, labels, label_weights, avg_factor=avg_factor
        )

    def compute_bbox_loss(
        self,
        bbox_pred: torch.Tensor,
        labels: torch.Tensor,
        bbox_targets: torch.Tensor,
        bbox_weights: torch.Tensor,
    ):
        pos_inds = labels > 0
        if self.reg_class_agnostic:
            pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
        else:
            pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[
                pos_inds, labels[pos_inds]
            ]
        return self.loss_bbox(
            pos_bbox_pred,
            bbox_targets[pos_inds],
            bbox_weights[pos_inds],
            avg_factor=bbox_targets.size(0),
        )

    def _add_conv_fc_branch(
        self, num_branch_convs, num_branch_fcs, in_channels, is_shared=False
    ):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels
                )
                branch_convs.append(
                    ConvModule(
                        conv_in_channels, self.conv_out_channels, 3, padding=1
                    )
                )
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (
                is_shared or self.num_shared_fcs == 0
            ) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels
                )
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels)
                )
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim


if __name__ == "__main__":
    ZSD = SemanticBBoxHead().cuda()
    x = torch.randn(1, 2048, 14, 14).cuda()
    bg = torch.randn(300).cuda()
    y = ZSD(x, bg)
