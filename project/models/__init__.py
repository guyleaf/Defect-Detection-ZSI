from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ..data import ImageAnnotation, ImageMetadata
from .zero_shot_mask_rcnn import ZeroShotMaskRCNN


class ZeroShotMaskModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, momentum: float = 0.9) -> None:
        super().__init__()
        self.model = ZeroShotMaskRCNN()
        self.lr = lr
        self.momentum = momentum

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch: tuple) -> STEP_OUTPUT:
        output = self(*batch)
        return

    def validation_step(self, batch: tuple) -> Optional[STEP_OUTPUT]:
        output = self(*batch)
        return

    def test_step(self, batch: tuple) -> Optional[STEP_OUTPUT]:
        output = self(*batch)
        return

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(self.momentum, 0.999)
        )
