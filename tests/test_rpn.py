import torch
from project.data.image import ImageMetadata
from project.models.ba_rpn import BackgroundAwareRPNHead, BackgroundAwareRPN

from project.models.utils.anchor import AnchorGenerator

if __name__ == "__main__":
    feature_maps = [torch.ones((10, 256, 14, 14)), torch.ones((10, 256, 28, 28))]
    images = torch.ones((10, 3, 224, 224))
    anchor_generator = AnchorGenerator(sizes=(4, 8), scales=(8,))
    anchors = anchor_generator(
        feature_maps,
        images,
    )
    print(anchors[0].shape)
    print(anchor_generator.num_anchors_per_size)

    head = BackgroundAwareRPNHead(
        in_channels=256,
        num_anchors=anchor_generator.num_anchors_per_size,
        vec_path="../data/keycap/class_w2v_with_bg.npy",
        voc_path="../data/keycap/vocabulary_w2v.npy",
    )
    output = head(feature_maps[0])
    print(output[0].shape)

    rpn = BackgroundAwareRPN(head=head, anchor_generator=anchor_generator)
    rpn.eval()
    metas = [ImageMetadata(name="test", size=tuple(img.shape[-2:])) for img in images]
    output = rpn(feature_maps, images, metas)
    print(output[0][0].shape)
