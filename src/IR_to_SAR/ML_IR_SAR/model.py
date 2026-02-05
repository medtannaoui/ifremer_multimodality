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
    conditional_model = False,
    cross_attention_dim=0,
    batch_size=None,
    cond_dim=16
):
    """
    Creates a UNet2DModel with the specified configuration.
    """
    if not conditional_model:
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
    else : 
        unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            dropout=dropout,
            norm_num_groups=16,
            cross_attention_dim=cross_attention_dim   #features diemnsion
        )
        model = ConditionalUNet(unet=unet, cond_dim=cond_dim,cross_attention_dim=cross_attention_dim)

    return model


class ConditionalUNet(nn.Module):
    def __init__(self, unet, cond_dim, cross_attention_dim):
        super().__init__()

        self.unet = unet
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cross_attention_dim*2),
            nn.SiLU(),
            nn.Linear(cross_attention_dim*2, cross_attention_dim),
        )

    def forward(self, x, timestep, cond):
        cond_embed = self.mlp(cond)      # (B, D)
        cond_embed = cond_embed.unsqueeze(1)  #-B,1,D)

        return self.unet(
            sample=x,
            timestep=timestep,
            encoder_hidden_states=cond_embed,
        )

