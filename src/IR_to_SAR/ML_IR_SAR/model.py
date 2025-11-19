## Thsi script will be used to crerate a diffusion model (U-NET)

import torch
import torch.nn as nn
from diffusers import UNet2DModel


def create_model(
    image_size=(304, 304),   # patch final utile
    in_channels=1,           # IR input (gray-scale)
    out_channels=1,          # SAR output (wind speed map)
    block_out_channels=(64, 128),  # multiscale feature depth
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
    ),
):
    """
    Creates a UNet2DModel with the specified configuration.
    """
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    return model

