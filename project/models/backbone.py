import torch
from torch import nn, Tensor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone



class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.backbone = resnet_fpn_backbone('resnet50', pretrained = True)

    def forward(self, x : Tensor) -> list:
        output = self.backbone(x)
        output = [v for k, v in output.items()] 
        return output

def test():
    extractor = Extractor()
    x = torch.randn(1, 3, 224, 224)
    y = extractor(x)
    
