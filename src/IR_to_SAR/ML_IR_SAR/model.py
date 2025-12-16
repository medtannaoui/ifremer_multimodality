## Thsi script will be used to crerate a diffusion model (U-NET)

import torch
import torch.nn as nn
from diffusers import UNet2DModel, UNet2DConditionModel


def create_model(
    image_size=(256, 256),   # patch final utile
    in_channels=1,           # IR 
    out_channels=1,          # SAR output (wind speed map)
    block_out_channels=(32,64,128,64),
    dropout= 0.2,
    down_block_types=(
        "DownBlock2D",
        # "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",   # block with attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
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




class IR2SARConditionedUNet(nn.Module):
    """
    UNet conditionné pour la régression IR → SAR
    """

    def __init__(
        self,
        image_size=(128, 128),
        in_channels=1,          # IR
        out_channels=1,         # SAR
        cond_dim=1,             
        block_out_channels=(32, 64, 128, 64),
        dropout=0.2,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ):
        super().__init__()

        self.unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            dropout=dropout,
            cross_attention_dim=cond_dim,  
        )

    def forward(self, x, cond):
        """
        x    : (B, C, H, W)   → IR
        cond : (B, cond_dim) → variables physiques (vmax, shear, ...)
        """

        cond = cond.unsqueeze(1) 
        out = self.unet(
            sample=x,
            timestep=0,                
            encoder_hidden_states=cond  
        )

        return out.sample