

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


@torch.no_grad()
def ode_solver_direct(model, z, ir_input, num_steps=50):
    """
    Euler ODE solver for the direct FM model.

    Args:
        model:     trained FM UNet (in_channels = C_IR + 1)
        z:         (B, 1, H, W) initial noise ~ N(0, I)
        ir_input:  (B, C_IR, H, W) IR conditioning image
        num_steps: Euler integration steps

    Returns:
        (B, 1, H, W) predicted SAR wind speed (in normalised space)
    """
    x_t = z.clone()
    dt  = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((x_t.shape[0],), i / num_steps, device=x_t.device)
        model_input   = torch.cat([x_t, ir_input], dim=1)
        pred_velocity = model(model_input, t).sample
        x_t           = x_t + pred_velocity * dt

    return x_t


@torch.no_grad()
def generate_ensemble(model, ir_input, n_members=20, num_steps=50, device=None):
    """
    Generate an ensemble of SAR predictions from a single IR input.

    Args:
        model:      trained FM UNet
        ir_input:   (1, C_IR, H, W) or (C_IR, H, W) — single sample
        n_members:  ensemble size
        num_steps:  Euler steps
        device:     if None, uses ir_input.device

    Returns:
        ensemble: (n_members, 1, H, W) tensor of predictions
    """
    if ir_input.ndim == 3:
        ir_input = ir_input.unsqueeze(0)

    assert ir_input.shape[0] == 1, "Pass a single sample (B=1)"
    device = device or ir_input.device

    # Broadcast IR conditioning to all ensemble members
    ir_batch = ir_input.to(device).expand(n_members, -1, -1, -1)   # (n, C_IR, H, W)

    # Independent noise for each member
    H, W = ir_input.shape[2], ir_input.shape[3]
    z = torch.randn(n_members, 1, H, W, device=device)

    # Single batched ODE solve
    ensemble = ode_solver_direct(model, z, ir_batch, num_steps)

    return ensemble   # (n_members, 1, H, W)

def compute_uncertainty_maps(ensemble):
    """
    Compute ensemble mean and std.

    Args:
        ensemble: (n_members, 1, H, W)

    Returns:
        mean: (1, H, W),  std: (1, H, W)
    """
    return ensemble.mean(0), ensemble.std(0)



def plot_ensemble_results(ir_input, sar_target, ensemble, stats, mask=None, save_path=None,cmap_sar=None, cmap_ir=None,save_pic=False,
                          title=""):
    """
    4-panel figure: IR | SAR target | FM mean | FM uncertainty.

    Args:
        ir_input:   (1, C_IR, H, W) or (C_IR, H, W)
        sar_target: (1, 1, H, W) or (H, W) — can be None (for blind prediction)
        ensemble:   (n_members, 1, H, W)
        stats:      dict with "mean" and "std" (SAR normalisation)
        mask:       (1, 1, H, W) or None
    """
    sar_mean_stat = stats["mean"]
    sar_std_stat  = stats["std"]

    if ir_input.ndim == 4:
        ir_input = ir_input[0]

    ens_mean, ens_std = compute_uncertainty_maps(ensemble)

    # Denormalise to physical units (m/s)
    ens_mean_phys = ens_mean[0].cpu().numpy() * sar_std_stat + sar_mean_stat
    ens_std_phys  = ens_std[0].cpu().numpy()  * sar_std_stat   # std scales with std


    if save_pic :
        n_panels = 4 if sar_target is not None else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

        # IR input (first channel)
        axes[0].imshow(ir_input[0].cpu().numpy().squeeze(), cmap=cmap_ir or "gray")
        axes[0].set_title("IR input (ch 0)")

        panel = 1
        if sar_target is not None:
            if sar_target.ndim == 4:
                sar_target = sar_target[0, 0]
            elif sar_target.ndim == 3:
                sar_target = sar_target[0]
            sar_phys = sar_target.cpu().numpy() * sar_std_stat + sar_mean_stat
            if mask is not None:
                if mask.ndim == 4: mask = mask[0, 0]
                sar_phys = np.where(mask.cpu().numpy() > 0, sar_phys, np.nan)
            im = axes[panel].imshow(sar_phys.squeeze(), cmap=cmap_sar or "RdBu_r",vmin=0,vmax=160/1.94384449)
            axes[panel].set_title("SAR target (m/s)")
            plt.colorbar(im, ax=axes[panel], fraction=0.046)
            panel += 1

        im = axes[panel].imshow(ens_mean_phys, cmap=cmap_sar or "RdBu_r",vmin=0,vmax=160/1.94384449)
        axes[panel].set_title(f"FM ensemble mean\n(n={ensemble.shape[0]})")
        plt.colorbar(im, ax=axes[panel], fraction=0.046)
        panel += 1

        im = axes[panel].imshow(ens_std_phys, cmap=cmap_sar or "hot",vmin=0,vmax=160/1.94384449)
        axes[panel].set_title("FM uncertainty (std, m/s)")
        plt.colorbar(im, ax=axes[panel], fraction=0.046)

        for ax in axes:
            ax.axis("off")

        if title:
            fig.suptitle(title, fontsize=12)
        
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        plt.tight_layout()  
    return ens_mean_phys, ens_std_phys


# rank histogram : verfiying that the true SAR value is statistically indistinguishable from the ensemble distribution
def rank_histogram(ensemble, sar_target, mask):
    """
    Compute the rank histogram (Talagrand diagram) for ensemble calibration.

    Args:
        ensemble:   (n_members, 1, H, W)
        sar_target: (1, 1, H, W)
        mask:       (1, 1, H, W) bool

    Returns:
        ranks: array of shape (n_valid_pixels,)
    """
    n = ensemble.shape[0]

    if sar_target.ndim == 3:
        sar_target = sar_target.unsqueeze(0)  # (1, 1, H, W)
    
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)  # (1, 1, H, W)
    
    if ensemble.ndim == 3:
        ensemble = ensemble.unsqueeze(1)  # (n_members, 1, H, W)

    valid = mask[0, 0].bool()   # (H, W)
    ens_flat = ensemble[:, 0][..., valid].T   # (n_valid, n_members)
    obs_flat = sar_target[0, 0][valid]        # (n_valid,)
    # Sort each ensemble and find rank of observation
    sorted_ens = torch.sort(ens_flat, dim=1).values  # (n_valid, n_members)
    ranks = (obs_flat.unsqueeze(1) > sorted_ens).sum(dim=1)  # (n_valid,) in [0, n]
    return ranks.cpu().numpy(), n


def plot_rank_histogram(ranks, n_members, title="",save_pth=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ranks, bins=n_members + 1, range=(-0.5, n_members + 0.5),
            density=True, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axhline(1.0 / (n_members + 1), color="red", linestyle="--", label="Uniform (ideal)")
    ax.set_xlabel("Rank of observation in ensemble")
    ax.set_ylabel("Relative frequency")
    ax.set_title(f"Rank histogram — {title}")
    ax.legend()
    plt.tight_layout()
    if save_pth:
        plt.savefig(save_pth, dpi=150)
        plt.close(fig)
    



def ode_solver_with_guidance(
    model,
    z,
    ir_input,
    num_steps,
    obs_sar,
    obs_mask,
    sar_mean_stat=0.0,
    sar_std_stat=1.0,
    guidance_strength=0.3,
    guidance_start_t=0.0,
    guidance_end_t=1.0,
):
    """
    Guided Euler ODE solver — steers samples toward sparse SAR observations.

    Args:
        model:              trained FM UNet
        z:                  (B, 1, H, W) initial noise
        ir_input:           (B, C_IR, H, W) IR conditioning
        num_steps:          Euler steps
        obs_sar:            (B, 1, H, W) observed SAR wind speed (normalised)
        obs_mask:           (B, 1, H, W) bool, True = observed pixel
        sar_mean_stat:      scalar — dataset mean used for normalisation
        sar_std_stat:       scalar — dataset std used for normalisation
        guidance_strength:  how strongly to pull toward observations (0 = no guidance)
        guidance_start_t:   apply guidance only for t >= start
        guidance_end_t:     apply guidance only for t <= end

    Returns:
        (B, 1, H, W) guided prediction
    """
    x_t = z.clone().requires_grad_(True).unsqueeze(1)  # (B, 1, H, W)
    dt  = 1.0 / num_steps

    for i in range(num_steps):
        t_val = i / num_steps
        t     = torch.full((x_t.shape[0],), t_val, device=x_t.device)
        

        model_input   = torch.cat([x_t, ir_input], dim=1)
        pred_velocity = model(model_input, t).sample

        apply_guidance = (
            guidance_strength > 0
            and guidance_start_t <= t_val <= guidance_end_t
            and obs_mask.any()
        )

        if apply_guidance:
            # Current prediction in normalised space
            current_pred = x_t

            # MSE between current prediction and observations at observed locations
            error = ((current_pred - obs_sar) ** 2 * obs_mask.float())
            # Normalise error to avoid scale dependence
            norm_error = error / (error.detach().mean() + 1e-8)
            loss = norm_error.sum()

            # Gradient w.r.t. x_t
            guidance_grad = torch.autograd.grad(loss, x_t, retain_graph=False)[0]

            # Clip at 95th percentile to avoid instability
            clip_val = torch.quantile(guidance_grad.abs(), 0.95)
            guidance_grad = guidance_grad.clamp(-clip_val, clip_val)

            # Time-weighted guidance: stronger toward end of trajectory
            # (samples become more structured as t increases)
            time_weight = t_val

            corrected_velocity = pred_velocity - guidance_strength * guidance_grad * time_weight
            x_t = (x_t + corrected_velocity * dt).detach().requires_grad_(True)
        else:
            x_t = (x_t + pred_velocity * dt).detach().requires_grad_(True)

    return x_t.detach()
    



@torch.no_grad()
def ode_solver_with_reprojection(
    model,
    z,
    ir_input,
    num_steps,
    obs_sar,
    obs_mask,
    reprojection_start_t=0.5,
    reprojection_strength=1.0,
):
    """
    ODE solver with hard reprojection at observation locations.
    """

    # Ensure shapes are (B,1,H,W)
    if z.ndim == 3:
        x_t = z.clone().unsqueeze(1)
    elif z.ndim == 4:
        x_t = z.clone()
    else:
        raise ValueError(f"Unexpected z shape: {z.shape}")

    if obs_sar.ndim == 3:
        obs_sar = obs_sar.unsqueeze(1)
    elif obs_sar.ndim != 4:
        raise ValueError(f"Unexpected obs_sar shape: {obs_sar.shape}")

    if obs_mask.ndim == 3:
        obs_mask = obs_mask.unsqueeze(1)
    elif obs_mask.ndim != 4:
        raise ValueError(f"Unexpected obs_mask shape: {obs_mask.shape}")

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = i / num_steps
        t = torch.full((x_t.shape[0],), t_val, device=x_t.device)

        model_input = torch.cat([x_t, ir_input], dim=1)
        print("x_t:", x_t.shape, "ir_input:", ir_input.shape, "model_input:", model_input.shape)

        pred_velocity = model(model_input, t).sample
        x_t = x_t + pred_velocity * dt

        if t_val >= reprojection_start_t and obs_mask.any():
            eps = 1e-6
            scale = obs_sar / (x_t + eps)

            alpha = min((t_val - reprojection_start_t) / (1.0 - reprojection_start_t + eps), 1.0)
            alpha *= reprojection_strength
            scale_applied = (1 - alpha) + alpha * scale

            x_t = torch.where(obs_mask.bool(), x_t * scale_applied, x_t)

    return x_t



@torch.no_grad()
def ode_solver_residual(fm_model, z, mean_pred, num_steps=50):   #1 example
    """
    Euler ODE solver in residual space.

    Args:
        fm_model:   residual FM UNet (in_channels=2)
        z:          (B, 1, H, W) initial noise (normalised residual space)
        mean_pred:  (B, 1, H, W) regression mean field (frozen)
        num_steps:  Euler steps

    Returns:
        (B, 1, H, W) predicted normalised residual x_1
    """
    x_t = z.clone()
    dt  = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((x_t.shape[0],), i / num_steps, device=x_t.device)
        model_input   = torch.cat([x_t, mean_pred], dim=1)
        pred_velocity = fm_model(model_input, t).sample
        x_t           = x_t + pred_velocity * dt

    return x_t   # normalised residual x_1

def reconstruct_from_residual(x_1_norm, mean_pred, residual_mean, residual_std):  #sar pred
    """
    Convert normalised residual back to physical wind speed.

    Args:
        x_1_norm:      (B, 1, H, W) normalised residual from ODE solver
        mean_pred:     (B, 1, H, W) regression mean field (in normalised SAR space)
        residual_mean: pre-computed residual mean (scalar)
        residual_std:  pre-computed residual std  (scalar)

    Returns:
        (B, 1, H, W) SAR prediction in normalised SAR space
    """
    residual = x_1_norm * residual_std + residual_mean
    return mean_pred + residual


def generate_residual_ensemble(
    fm_model, mean_pred, n_members=20, num_steps=50,
    residual_mean=0.0, residual_std=1.0,
):
    """
    Generate ensemble of SAR predictions using residual FM.

    Args:
        fm_model:       residual FM UNet
        mean_pred:      (1, 1, H, W) regression mean field
        n_members:      ensemble size
        num_steps:      Euler steps
        residual_mean:  float
        residual_std:   float

    Returns:
        ensemble: (n_members, 1, H, W)
    """
    device = mean_pred.device
    H, W   = mean_pred.shape[2], mean_pred.shape[3]

    # Broadcast mean_pred to all ensemble members
    mean_batch = mean_pred.expand(n_members, -1, -1, -1)   # (n, 1, H, W)
    z          = torch.randn(n_members, 1, H, W, device=device)

    with torch.no_grad():
        x_1_norm = ode_solver_residual(fm_model, z, mean_batch, num_steps)
        ensemble = reconstruct_from_residual(x_1_norm, mean_batch, residual_mean, residual_std)

    return ensemble   # (n_members, 1, H, W)