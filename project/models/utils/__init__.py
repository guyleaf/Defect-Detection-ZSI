from functools import partial
from typing import Union

import numpy as np
import torch

from .anchor import AnchorGenerator
from .bbox import BoxCoder, Matcher
from .rpn import concat_box_prediction_layers

from .conv_module import ConvModule, build_conv_layer
from .conv_ws import ConvWS2d, conv_ws_2d
from .norm import build_norm_layer
from .mask_utils import  mask_target_single, mask_target
from .bbox_utils import  bbox_target


def load_word_vectors(
    path: str, as_torch: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    data = np.load(path)
    if as_torch:
        data = torch.from_numpy(data)
    return data


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    https://github.com/open-mmlab/mmdetection/blob/df28da98926bc410e16bed1e9fc7d425d9a89495/mmdet/core/utils/misc.py#L11

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class BalancedPositiveNegativeSampler:
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(
        self, batch_size_per_image: int, positive_fraction: float
    ) -> None:
        """
        Args:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(
        self, matched_idxs: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.
        Returns:
            pos_idx (list[torch.Tensor])
            neg_idx (list[torch.Tensor])
        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[
                :num_pos
            ]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[
                :num_neg
            ]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx


__all__ = [
    "AnchorGenerator",
    "load_word_vectors",
    "multi_apply",
    "concat_box_prediction_layers",
    "BoxCoder",
    "Matcher",
    "BalancedPositiveNegativeSampler",
    "conv_ws_2d",
    "ConvWS2d",
    "build_conv_layer",
    "ConvModule",
    "mask_utils",
    "mask_target_single",
    "mask_target",
    "bbox_utils",
    "bbox_target",
    "build_norm_layer",
    "xavier_init",
    "normal_init",
    "uniform_init",
    "kaiming_init",
    "bias_init_with_prob",
    "Scale",
]
