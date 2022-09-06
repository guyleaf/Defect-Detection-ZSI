from math import ceil
import os
from typing import Optional

import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from project.data import ImageAnnotation, ImageMetadata

DEFAULT_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


class KeycapDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_file: Optional[str] = None,
        transforms: transforms.Compose = DEFAULT_TRANSFORMS,
    ) -> None:
        super().__init__()
        self._img_dir = img_dir
        self._imgs = os.listdir(img_dir)
        self._transforms = transforms
        self._ann_loaded = False
        if ann_file:
            self._ann_loaded = True
            self._load_annotations(ann_file)

    def _get_img_name(self, idx: int) -> str:
        if self._ann_loaded:
            img_name = self._img_infos[idx]["file_name"]
        else:
            img_name = self._imgs[idx]
        return img_name

    def _load_annotations(self, ann_file: str):
        self._coco = COCO(ann_file)
        self._cat_ids = self._coco.getCatIds()
        self._cat2label = {
            cat_id: i + 1 for i, cat_id in enumerate(self._cat_ids)
        }
        img_ids = self._coco.getImgIds()
        self._img_infos = self._coco.loadImgs(img_ids)

    def _get_img_and_metadata(
        self, idx: int
    ) -> tuple[Image.Image, ImageMetadata]:
        img_name = self._get_img_name(idx)
        img_path = os.path.join(self._img_dir, img_name)

        img = Image.open(img_path)

        img_metadata = ImageMetadata(name=img_name, size=tuple(img.size[::-1]))
        return img, img_metadata

    def _convert_mask_into_binary(
        self, ann_info: list[dict]
    ) -> list[torch.Tensor]:
        masks = []
        for ann in ann_info:
            mask = self._coco.annToMask(ann)
            masks.append(torch.from_numpy(mask))
        return masks

    def _parse_annotation(self, ann_info: list[dict]) -> ImageAnnotation:
        gt_bboxes = []
        gt_labels = []

        for ann in ann_info:
            x1, y1, w, h = ann["bbox"]
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get("iscrowd", False):
                continue

            gt_bboxes.append(bbox)
            gt_labels.append(self._cat2label[ann["category_id"]])

        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)
        gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
        gt_masks = self._convert_mask_into_binary(ann_info)

        return ImageAnnotation(
            bboxes=gt_bboxes, labels=gt_labels, masks=gt_masks
        )

    def _get_annotation(self, idx: int) -> ImageAnnotation:
        # default instance
        annotation = ImageAnnotation(
            bboxes=torch.zeros((0, 4), dtype=torch.float32),
            labels=torch.empty(0, dtype=torch.int64),
            masks=torch.empty(0, dtype=torch.int64),
        )
        if self._ann_loaded:
            img_id = self._img_infos[idx]["id"]
            ann_ids = self._coco.getAnnIds(imgIds=[img_id])
            ann_info = self._coco.loadAnns(ann_ids)
            annotation = self._parse_annotation(ann_info)
        return annotation

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, idx: int):
        img, metadata = self._get_img_and_metadata(idx)
        annotation = self._get_annotation(idx)

        img = self._transforms(img)
        return img, metadata, annotation


class KeycapDataModule(LightningDataModule):
    def __init__(
        self, root_dir: str, batch_size: int = 64, train_val_ratio: float = 0.9
    ) -> None:
        super().__init__()
        self._root_dir = root_dir
        self._batch_size = batch_size
        self._train_val_ratio = train_val_ratio

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            img_dir = os.path.join(self._root_dir, "train_seen")
            ann_file = os.path.join(
                self._root_dir, "annotations", "train_seen.json"
            )
            keycap_full = KeycapDataset(img_dir=img_dir, ann_file=ann_file)
            num_imgs = len(keycap_full)
            train_num_imgs = ceil(num_imgs * self._train_val_ratio)

            self.keycap_train, self.keycap_val = random_split(
                keycap_full, [train_num_imgs, num_imgs - train_num_imgs]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            img_dir = os.path.join(self._root_dir, "test_unseen")
            self.keycap_test = KeycapDataset(img_dir=img_dir)

    def train_dataloader(self):
        return DataLoader(self.keycap_train, batch_size=self._batch_size)

    def val_dataloader(self):
        return DataLoader(self.keycap_val, batch_size=self._batch_size)

    def test_dataloader(self):
        return DataLoader(self.keycap_test, batch_size=self._batch_size)


if __name__ == "__main__":
    img_dir = (
        "E:\\Git\\Defect-Detection-ZSI\\tests\\datasets\\keycap\\train_seen"
    )
    ann_file = "E:\\Git\\Defect-Detection-ZSI\\tests\\datasets\\keycap\\annotations\\train_seen.json"
    dataset = KeycapDataset(img_dir=img_dir, ann_file=ann_file)
    print(dataset[0])
    img_dir = (
        "E:\\Git\\Defect-Detection-ZSI\\tests\\datasets\\keycap\\test_unseen"
    )
    dataset = KeycapDataset(img_dir=img_dir)
    print(dataset[0])
