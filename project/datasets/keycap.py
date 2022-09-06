import json
import os
from typing import Callable, Optional

import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

DEFAULT_TRANSFORMS = transforms.Compose([])


class KeycapDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        ann_file: Optional[str] = None,
        transforms: transforms.Compose = DEFAULT_TRANSFORMS,
        ann_transforms: Callable = None
    ) -> None:
        super().__init__()
        self._image_dir = image_dir
        self._num_images = len(os.listdir(image_dir))
        self._transforms = transforms
        self._ann_transforms = ann_transforms
        if ann_file:
            with open(ann_file, "r") as f:
                self._image_anns = json.load(f)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        img_path = os.path.join(self._image_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class KeycapDataModule(LightningDataModule):
    def __init__(self, root_dir: str) -> None:
        super().__init__()
