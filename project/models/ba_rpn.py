from typing import Optional

import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation

from .utils import AnchorGenerator, load_word_vectors, multi_apply


class BackgroundAwareRPNHead(nn.Module):
    _voc: Optional[torch.Tensor]

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        vec_path: str,
        semantic_channels: int = 300,
        voc_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        # in source code: padding = 1
        self.conv = Conv2dNormActivation(
            in_channels, in_channels, 3, norm_layer=None
        )

        vec, voc = self._load_word_vectors(vec_path, voc_path)
        bg_vec = vec[:, 0].unsqueeze(-1).unsqueeze(-1)

        # visual feature -> semantic_encoder -> semantice feature -> cls_logits
        self.semantic_encoder = nn.Conv2d(in_channels, semantic_channels, 1)
        self.cls_logits = nn.Conv2d(semantic_channels, num_anchors * 2, 1)

        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

        self.register_buffer("_voc", voc, persistent=False)

        # initialize weights
        self.apply(self._init_weights)
        self._init_cls_logits(bg_vec)

    @torch.no_grad()
    def _init_cls_logits(self, bg_vec: torch.Tensor):
        weights = self.cls_logits.weight
        weights.data[::2] = bg_vec.expand(weights.size(0) // 2, -1)

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
        if self._voc:
            semantic_feature = torch.mm(semantic_feature, self._voc)
        cls_score = self.cls_logits(semantic_feature)

        # regression branch
        bbox_pred = self.bbox_pred(x)

        bg_vec = torch.mean(self.cls_logits.weight.data[::2], dim=0)
        return cls_score, bbox_pred, bg_vec


class BackgroundAwareRPN(nn.Module):
    head: nn.Module
    anchor_generator: AnchorGenerator

    def __init__(
        self, head: nn.Module, anchor_generator: AnchorGenerator
    ) -> None:
        self.head = head
        self.anchor_generator = anchor_generator

    def forward(self, feature_maps: list[torch.Tensor], images: torch.Tensor):
        cls_scores, bbox_preds, bg_vecs = multi_apply(self.head, feature_maps)
        anchors = self.anchor_generator(feature_maps, images)
