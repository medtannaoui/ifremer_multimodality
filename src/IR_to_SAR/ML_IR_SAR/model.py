## Thsi script will be used to crerate a diffusion model (U-NET)

import torch
import torch.nn as nn
from diffusers import UNet2DModel, UNet2DConditionModel
import numpy as np


def create_model(
    cfg,
    conditional_model=False,
    in_channels = None
):
    """
    Creates a UNet2DModel with the specified configuration.
    """
    if not conditional_model:
        model = UNet2DModel(
            sample_size=cfg.img_size,
            in_channels= in_channels if in_channels is not None else cfg.irwin_channels + (1 if cfg.add_era5 else 0),  # IR channels + optional ERA5 channel
            out_channels=cfg.out_channels,
            block_out_channels=cfg.block_out_channels,
            down_block_types=cfg.down_block_types,
            up_block_types=cfg.up_block_types,
            dropout=cfg.dropout,
            norm_num_groups=np.min([cfg.block_out_channels])
        )
    else : 
        unet = UNet2DConditionModel(
            sample_size=cfg.img_size,
            in_channels=cfg.inrwin_channels if cfg.add_era5 else cfg.irwin_channels +1,
            out_channels=cfg.out_channels,
            block_out_channels=cfg.block_out_channels,
            down_block_types=cfg.down_block_types,
            up_block_types=cfg.up_block_types,
            dropout=cfg.dropout,
            norm_num_groups=np.min([cfg.block_out_channels]),
            cross_attention_dim=cfg.cross_attention_dim   #features diemnsion
        )
        model = ConditionalUNet(unet=unet, cond_dim=cfg.cond_dim,cross_attention_dim=cfg.cross_attention_dim)
    return model

def create_fm_model_direct(cfg, in_channels_ir):
    """
    Create a UNet2DModel for direct flow matching.

    The only difference from the deterministic model:
      in_channels = in_channels_ir + 1   (extra channel for x_t)
      out_channels = 1                   (velocity in SAR space)
    """
    in_channels_fm = in_channels_ir + 1   # x_t (1 ch) + IR (C_ir ch)
    return create_model(
        cfg=cfg,
        conditional_model=False,
        in_channels=in_channels_fm
            )


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

#Use Residu 
def load_regression_model(checkpoint_path, cfg, 
                           device = "cpu"):
    """
    Load the existing regression model from best_regression_model.pt.
    Returns it frozen (requires_grad=False).
    """
    model = create_model(
        in_channels=cfg.in_channels,
        cfg=cfg,
        conditional_model=False,

    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)

    # Freeze all parameters — this model will NOT be trained
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model.to(device)

def create_fm_residual_model(cfg):
    """
    Create FM model for residual learning.

    in_channels = 2:
      channel 0 = x_t   (noisy normalised residual)
      channel 1 = mean_prediction  (regression output, the conditioning signal)
    out_channels = 1: predicted velocity in residual space
    """
    return create_model(
        in_channels=2,         # x_t_resid (1) + mean_pred (1)
        cfg=cfg,
        conditional_model=False,
    )
