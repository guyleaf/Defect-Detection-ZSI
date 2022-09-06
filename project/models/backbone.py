import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import collections



backbone = resnet_fpn_backbone('resnet50', pretrained = True)

def test(backbone):
    extractor = backbone
    x = torch.randn(1, 3, 224, 224)
    y = extractor(x)
    for k, v in y.items():
        print(k, v.shape)