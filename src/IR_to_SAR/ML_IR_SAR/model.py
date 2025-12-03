## Thsi script will be used to crerate a diffusion model (U-NET)

import torch
import torch.nn as nn
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler


def create_model(
    image_size=(256, 256),   # patch final utile
    in_channels=1,           # IR 
    out_channels=1,          # SAR output (wind speed map)
    block_out_channels=(64,64,128),
    dropout= 0.2,
    down_block_types=(
        "DownBlock2D",
        # "DownBlock2D",
        # "DownBlock2D",
        "AttnDownBlock2D",   # block with attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        # "UpBlock2D",
        # "UpBlock2D",
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
        dropout=dropout
    )
    return model

