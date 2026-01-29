## Thsi script will be used to crerate a diffusion model (U-NET)

import torch
import torch.nn as nn
from diffusers import UNet2DModel, UNet2DConditionModel


def create_model(
    image_size=None,   # patch final utile
    in_channels=1,           # IR 
    out_channels=1,          # SAR output (wind speed map)
    block_out_channels=None,
    dropout= 0.0,
    down_block_types=None,
    up_block_types=None,
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
        dropout=dropout,
        norm_num_groups=16
    )
    return model


