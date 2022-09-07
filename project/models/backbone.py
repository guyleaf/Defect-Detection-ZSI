from typing import Optional
import torch
from torch import nn, Tensor

import torchvision.models as models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class Extractor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet101",
        trainable_layers: int = 3,
        returned_layers: Optional[list[int]] = None
    ):
        super(Extractor, self).__init__()
        if backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
        elif backbone_name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT
        elif backbone_name == "resnet152":
            weights = models.ResNet152_Weights.DEFAULT
        else:
            raise ValueError(
                f"The backbone {backbone_name} is not implemented."
            )

        self.backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            weights=weights,
            trainable_layers=trainable_layers,
            returned_layers=returned_layers,
        )

    @property
    def out_channels(self) -> int:
        return self.backbone.out_channels

    def forward(self, x: Tensor) -> list:
        output = self.backbone(x)
        output = list(output.values())
        return output


if __name__ == "__main__":
    with torch.no_grad():
        extractor = Extractor(backbone_name="resnet34")
        # x = torch.randn(1, 3, 224, 224)
        x = torch.randn(1, 3, 500, 500)
        y = extractor(x)
        print(len(y))
