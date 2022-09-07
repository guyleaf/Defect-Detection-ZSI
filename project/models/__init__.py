from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from project.data import ImageAnnotation, ImageMetadata
from project.models.zero_shot_mask_rcnn import ZeroShotMaskRCNN


__all__ = ["ZeroShotMaskModel"]


class ZeroShotMaskModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, momentum: float = 0.9, **kwargs) -> None:
        super().__init__()
        self.model = ZeroShotMaskRCNN(**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.save_hyperparameters()

    def forward(self, args):
        return self.model(*args)

    def training_step(self, batch: tuple) -> STEP_OUTPUT:
        output = self(batch)
        return

    def validation_step(self, batch: tuple) -> Optional[STEP_OUTPUT]:
        # TODO: provide loss output
        output = self(batch)
        return

    def test_step(self, batch: tuple) -> Optional[STEP_OUTPUT]:
        img_metas = batch[1]
        bboxes, labels, masks, _ = self(batch)
        
        return

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(self.momentum, 0.999)
        )
