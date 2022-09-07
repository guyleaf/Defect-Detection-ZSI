from typing import Union

import torch
import torch.nn as nn


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple scales and ratios
    per feature map. This module assumes ratio = height / width for
    each anchor.

    and AnchorGenerator will output a set of len(sizes) * len(scales) * len(ratios) anchors
    per spatial location for each feature map i.

    Args:
        sizes (List[int] | Tuple[int]):
        scales (List[int] | Tuple[int]):
        ratios (List[float] | Tuple[float]):
    """

    _num_anchors: int
    _cell_anchors: list[torch.Tensor]

    def __init__(
        self,
        sizes: Union[list[int], tuple[int]] = (4, 8, 16, 32, 64),
        scales: Union[list[int], tuple[int]] = (8, 16, 32),
        ratios: Union[list[float], tuple[float]] = (0.5, 1.0, 2.0),
    ) -> None:
        super().__init__()

        if not isinstance(sizes, (list, tuple)):
            sizes = (sizes,)
        if not isinstance(scales, (list, tuple)):
            scales = (scales,)
        if not isinstance(ratios, (list, tuple)):
            ratios = (ratios,)

        self._num_anchors_per_size = len(scales) * len(ratios)
        self._cell_anchors = [
            self._generate_anchors(size, scales, ratios)
            for size in range(len(sizes))
        ]

    # For every (ratios, scales) combination, output a zero-centered anchor with those values.
    # This method assumes aspect ratio = height / width for an anchor.
    def _generate_anchors(
        self, base_size: int, scales: list[int], ratios: list[float]
    ) -> torch.Tensor:
        scales = torch.tensor(scales)
        ratios = torch.tensor(ratios)

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios

        ws = (base_size * w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (base_size * h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    @property
    def num_anchors_per_size(self) -> int:
        return self._num_anchors_per_size

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def _grid_anchors(
        self,
        grid_sizes: list[list[int]],
        strides: list[list[torch.Tensor]],
        device: Union[str, torch.device] = "cpu",
    ) -> list[torch.Tensor]:
        anchors: list[torch.Tensor] = []
        assert (
            len(grid_sizes) == len(strides) == len(self._cell_anchors)
        ), """Anchors should be Tuple[Tuple[int]] because each feature "
            map could potentially have different sizes and aspect ratios.
            There needs to be a match between the number of
            feature maps passed and the number of sizes / aspect ratios specified."""

        for size, stride, base_anchors in zip(
            grid_sizes, strides, self._cell_anchors
        ):
            base_anchors = base_anchors.to(device)
            grid_height, grid_width = size
            stride_height, stride_width = stride

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = stride_width * torch.arange(
                0, grid_width, dtype=torch.int32, device=device
            )
            shifts_y = stride_height * torch.arange(
                0, grid_height, dtype=torch.int32, device=device
            )
            shift_y, shift_x = torch.meshgrid(
                shifts_y, shifts_x, indexing="ij"
            )
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(
                    -1, 4
                )
            )

        return anchors

    def forward(
        self, feature_maps: list[torch.Tensor], images: torch.Tensor
    ) -> list[torch.Tensor]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_height, image_width = images.shape[-2:]
        device = feature_maps[0].device
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(
                    image_height // grid_height
                ),
                torch.empty((), dtype=torch.int64, device=device).fill_(
                    image_width // grid_width
                ),
            ]
            for grid_height, grid_width in grid_sizes
        ]

        anchors_over_all_feature_maps = self._grid_anchors(
            grid_sizes, strides, device
        )
        anchors: list[torch.Tensor] = []
        for _ in range(images.size(0)):
            anchors_in_image = torch.concat(
                anchors_over_all_feature_maps, dim=0
            )
            anchors.append(anchors_in_image)
        return anchors
