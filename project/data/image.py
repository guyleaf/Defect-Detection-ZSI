from typing import Union
import torch


class ImageMetadata:
    """
    Metadata info of an image

    Args:
        name (str): the name of image
        shape (tuple[int, int, int]): the shape of image (C, H, W)
    """

    def __init__(self, name: str, shape: tuple[int, int, int]) -> None:
        self._name = name
        self._shape = shape

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape


class ImageAnnotation:
    """
    Annotation info of an image

    Args:
        bboxes (torch.Tensor): the ground truth bounding boxes
        labels (torch.Tensor): the ground truth labels
        masks (torch.Tensor): the ground truth masks
    """

    def __init__(
        self, bboxes: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
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
    def masks(self) -> torch.Tensor:
        return self._masks

    def to(self, device: Union[str, torch.device]) -> None:
        self._bboxes = self._bboxes.to(device)
        self._labels = self._labels.to(device)
        self._masks = self._masks.to(device)
