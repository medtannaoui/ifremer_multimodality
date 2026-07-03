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
    model = create_model(
        in_channels=cfg.in_channels,
        cfg=cfg,
        conditional_model=False,

    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def create_fm_residual_model(cfg):
    return create_model(
        in_channels=2,         # x_t_resid (1) + mean_pred (1)
        cfg=cfg,
        conditional_model=False,
    )

def apply_random_channel_dropout(
    x,
    drop_prob=0.3,
    min_keep_channels=4,
    protect_channels=None,
):
    if drop_prob <= 0.0:
        return x
    B, C, H, W = x.shape
    device = x.device
    if protect_channels is None:
        protect_channels = []
    x_out = x.clone()
    for b in range(B):
        keep_mask = torch.ones(C, dtype=torch.bool, device=device)
        droppable = [c for c in range(C) if c not in protect_channels]
        if len(droppable) == 0:
            continue
        rand_mask = torch.rand(len(droppable), device=device) > drop_prob
        for i, c in enumerate(droppable):
            keep_mask[c] = rand_mask[i]
        for c in protect_channels:
            keep_mask[c] = True
        if keep_mask.sum() < min_keep_channels:
            missing = min_keep_channels - int(keep_mask.sum().item())
            dropped_candidates = [c for c in droppable if not keep_mask[c]]
            if len(dropped_candidates) > 0:
                perm = torch.randperm(len(dropped_candidates), device=device)
                for idx in perm[:missing]:
                    keep_mask[dropped_candidates[int(idx.item())]] = True
        x_out[b, ~keep_mask, :, :] = 0.0
    return x_out