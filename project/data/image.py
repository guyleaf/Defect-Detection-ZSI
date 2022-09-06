from typing import Union
import torch


class ImageMetadata:
    """
    Metadata info of an image

    Args:
        name (str): the name of image
        size (tuple[int, int]): the size of image (H, W)
    """

    def __init__(self, name: str, size: tuple[int, int]) -> None:
        self._name = name
        self._size = size

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> tuple[int, int]:
        return self._size


class ImageAnnotation:
    """
    Annotation info of an image

    Args:
        bboxes (torch.Tensor): the ground truth bounding boxes
        labels (torch.Tensor): the ground truth labels
        masks (torch.Tensor): the ground truth masks
    """

    def __init__(
        self, bboxes: torch.Tensor, labels: torch.Tensor, masks: list[torch.Tensor]
    ) -> None:
        self._bboxes = bboxes
        self._labels = labels
        self._masks = masks

    @property
    def bboxes(self) -> torch.Tensor:
        return self._bboxes

    @property
    def labels(self) -> torch.Tensor:
        return self.labels

    @property
    def masks(self) -> list[torch.Tensor]:
        return self._masks

    def to(self, device: Union[str, torch.device]) -> None:
        self._bboxes = self._bboxes.to(device)
        self._labels = self._labels.to(device)
        self._masks = self._masks.to(device)
