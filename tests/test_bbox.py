import torch
from project.models.semantic_bbox_head import SemanticBBoxHead

if __name__ == "__main__":
    ZSD = SemanticBBoxHead(
        "E:\\Git\\Defect-Detection-ZSI\\data\\keycap\\class_w2v_with_bg.npy",
        "E:\\Git\\Defect-Detection-ZSI\\data\\keycap\\vocabulary_w2v.npy",
    )
    x = torch.randn(1, 2048, 14, 14)
    bg = torch.randn(300)
    y = ZSD(x, bg)
