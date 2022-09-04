from functools import partial
from typing import Union

import numpy as np
import torch


def load_word_vectors(path: str, delimiter: str = ",", as_torch: bool = False) -> Union[np.ndarray, torch.Tensor]:
    data = np.loadtxt(path, dtype=np.float32, delimiter=delimiter)
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
