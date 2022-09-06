import torch
from torch import nn, Tensor

from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=ResNet50_Weights.DEFAULT)

    def forward(self, x: Tensor) -> list:
        output = self.backbone(x)
        output = list(output.values())
        return output


if __name__ == "__main__":
    with torch.no_grad():
        extractor = Extractor()
        x = torch.randn(1, 3, 224, 224)
        y = extractor(x)
        print(len(y))
