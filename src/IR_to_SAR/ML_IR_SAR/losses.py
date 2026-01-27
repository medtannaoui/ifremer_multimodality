# this script will be sued for create losses sued in train
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional import structural_similarity_index_measure as ssim



def pix2pix_l1_loss(sar_valid, pred_valid, mask):

    if sar_valid.ndim == 3:
        sar_valid = sar_valid.unsqueeze(1)  
                
    if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    
    loss_mse = F.l1_loss(pred_valid*mask, sar_valid*mask)
    return loss_mse

def gradient_l1_loss(sar_valid,pred_valid,mask):
       
    if sar_valid.ndim == 3:
        sar_valid = sar_valid.unsqueeze(1)  
                
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    
    gx_sar, gy_sar = torch.gradient(sar_valid, dim=(2,3))
    gx_pred, gy_pred = torch.gradient(pred_valid, dim=(2,3))

    loss_grad = (
            F.l1_loss(gx_pred * mask, gx_sar * mask) +
            F.l1_loss(gy_pred * mask, gy_sar * mask)
        )
    return loss_grad

def build_radius_map(H, W, device):
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    r = torch.sqrt((x - cx)**2 + (y - cy)**2)
    return r

def radial_profile_batch(sar, r_map, r_bins):
    """
    sar: (B,1,H,W)
    r_map: (H,W)
    returns: (B, R)
    """
    B = sar.shape[0]
    profiles = []

    for i in range(B):
        v = sar[i, 0]
        prof = []
        for r in r_bins:
            mask = (r_map >= r) & (r_map < r + 1)
            prof.append(v[mask].mean())
        profiles.append(torch.stack(prof))

    return torch.stack(profiles) 

def radial_vmax_l1_loss(sar_valid, pred_valid, mask, r_bins=None):
    
    if sar_valid.ndim == 3:
        sar_valid = sar_valid.unsqueeze(1)

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    B, _, H, W = sar_valid.shape
    device = sar_valid.device

    r_map = build_radius_map(H, W, device)

    if r_bins is None:
        r_max = int(min(H, W) // 2)
        r_bins = torch.arange(0, r_max, device=device)

    R = len(r_bins)
    prof_true = torch.zeros((B, R), device=device)
    prof_pred = torch.zeros((B, R), device=device)

    for b in range(B):
        v_true = sar_valid[b, 0]
        v_pred = pred_valid[b, 0]
        m = mask[b, 0]

        for i, r in enumerate(r_bins):
            ring = (r_map >= r) & (r_map < r + 1) & (m > 0)

            if ring.sum() > 0:
                prof_true[b, i] = v_true[ring].mean()
                prof_pred[b, i] = v_pred[ring].mean()
            else:
                if i > 0:
                    prof_true[b, i] = prof_true[b, i - 1]
                    prof_pred[b, i] = prof_pred[b, i - 1]
                else:
                    prof_true[b, i] = 0
                    prof_pred[b, i] = 0

    return F.l1_loss(prof_pred, prof_true)

    

def combined_sar_loss(
    sar_valid,
    pred_valid,
    mask,
    w_pix=1.0,
    w_grad=0.0,
    w_radial=0.0,
    r_bins=None,
    bin_edges=None,
    bin_weights=None,
    use_weighted_pix=False,
):
    loss = 0.0
    loss_dict = {}
    l_radial, l_grad, l_pix = 0.0, 0.0, 0.0

    s = w_pix + w_grad + w_radial
    s = s if s > 0 else 1.0

    if w_pix > 0:
        if use_weighted_pix:
            assert bin_edges is not None and bin_weights is not None
            wmap = make_weight_map(sar_valid, mask, bin_edges, bin_weights)
            l_pix = weighted_l1(pred_valid, sar_valid, mask, wmap)
        else:
            l_pix = pix2pix_l1_loss(sar_valid, pred_valid, mask)

        loss += w_pix * l_pix
        loss_dict["loss_pix"] = l_pix.detach()

    if w_grad > 0:
        l_grad = gradient_l1_loss(sar_valid, pred_valid, mask)
        loss += w_grad * l_grad
        loss_dict["loss_grad"] = l_grad.detach()

    if w_radial > 0:
        l_radial = radial_vmax_l1_loss(sar_valid, pred_valid, mask, r_bins=r_bins)
        loss += w_radial * l_radial
        loss_dict["loss_radial"] = l_radial.detach()

    loss = loss / s
    loss_dict["loss_total"] = loss.detach()

    return loss, l_pix.detach().cpu() if w_pix>0 else l_pix, l_grad.detach().cpu() if w_grad>0 else l_grad, l_radial.detach().cpu() if w_radial >0 else l_radial

@torch.no_grad()
def compute_bin_weights_from_loader(
    train_loader,
    bin_edges,
    device,
    alpha=0.5,     # 0.5 doux, 1.0 inverse fréquence strict
    eps=1e-6,
    max_batches=None
):
    num_bins = bin_edges.numel() - 1
    counts = torch.zeros(num_bins, device=device, dtype=torch.float64)

    for b, (x, sar, mask, _) in enumerate(train_loader):
        if max_batches is not None and b >= max_batches:
            break

        sar = sar.to(device)
        mask = mask.to(device)
        sar = torch.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)

        valid = mask > 0.5
        if valid.sum() == 0:
            continue

        v = sar[valid].float()
        idx = torch.bucketize(v, bin_edges, right=False) - 1
        idx = idx.clamp(0, num_bins - 1)

        counts += torch.bincount(idx, minlength=num_bins).to(torch.float64)

    probs = counts / counts.sum().clamp_min(1.0)

    weights = 1.0 / torch.pow(probs + eps, alpha)
    weights = weights / weights.mean().clamp_min(1e-12)  # stabilise

    return weights.to(torch.float32), probs.to(torch.float32), counts

@torch.no_grad()
def compute_bin_edges_quantiles(
    train_loader,
    device,
    num_bins=5,
    max_batches=None,
    eps=1e-6,
):
    """
    Calcule bin_edges (num_bins+1,) automatiquement avec min/max sur les pixels valides (mask)
    sans stocker tous les pixels.
    """
    vmin = None
    vmax = None

    for b, (x, sar, mask, _) in enumerate(train_loader):
        if max_batches is not None and b >= max_batches:
            break

        sar = sar.to(device)
        mask = mask.to(device)

        sar = torch.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)
        valid = mask > 0.5
        if valid.sum() == 0:
            continue

        v = sar[valid].float()
        bmin = v.min()
        bmax = v.max()

        vmin = bmin if vmin is None else torch.minimum(vmin, bmin)
        vmax = bmax if vmax is None else torch.maximum(vmax, bmax)

    if vmin is None or vmax is None:
        raise RuntimeError("No valid pixels found to compute min/max bin edges.")

    # Evite vmin == vmax
    if torch.isclose(vmin, vmax):
        vmax = vmin + eps

    edges = torch.linspace(vmin, vmax, steps=num_bins + 1, device=device)
    return edges


def make_weight_map(target, mask, bin_edges, bin_weights): 
    if target.ndim == 3: target = target.unsqueeze(1)
    if mask.ndim == 3: mask = mask.unsqueeze(1)

    device = target.device
    bin_edges = bin_edges.to(device)
    bin_weights = bin_weights.to(device)

    idx = torch.bucketize(target, bin_edges, right=False) - 1
    idx = idx.clamp(0, bin_weights.numel() - 1)

    wmap = bin_weights[idx] * mask
    return wmap


def weighted_l1(pred, target, mask, wmap, eps=1e-8):
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if target.ndim == 3: target = target.unsqueeze(1)
    if mask.ndim == 3: mask = mask.unsqueeze(1)

    diff = torch.abs(pred - target)
    num = (diff * wmap).sum()
    den = wmap.sum().clamp_min(eps)
    return num / den





######################
#"Article Loss"
########################
def build_radius_map(H, W, device):
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    r = torch.sqrt((xx - cx)**2 + (yy - cy)**2)  # en pixels
    return r


def make_weight_map_wspd_rad(
    target, mask,
    vmax_train_max=1.0,
    rmw_train_min=1.0,
    r_map=None,
    eps=1e-8
):
    """
    Retourne wmap = Wwspd * Wrad * mask
    target: (B,1,H,W) ou (B,H,W)
    mask:   (B,1,H,W) ou (B,H,W)
    r_map:  (H,W) rayon (pixels ou km) cohérent avec rmw_train_min
    """
    if target.ndim == 3: target = target.unsqueeze(1)
    if mask.ndim == 3: mask = mask.unsqueeze(1)

    B, _, H, W = target.shape
    device = target.device

    if r_map is None:
        r_map = build_radius_map(H, W, device)  # pixels
    r = r_map[None, None, :, :]  # (1,1,H,W)

    vmax_train_max = torch.tensor(vmax_train_max, device=device, dtype=target.dtype)
    rmw_train_min  = torch.tensor(rmw_train_min,  device=device, dtype=target.dtype).clamp_min(eps)

    # Wwspd = 1 + (T - Vmax_train_max)^2 / 9
    Wwspd = 1.0 + ((target - vmax_train_max) ** 2) / 9.0

    # Wrad = (1 + r / (RMW_min^2))^-1
    Wrad = 1.0 / (1.0 + (r / rmw_train_min ** 2))

    wmap = Wwspd * Wrad * mask
    return wmap


def weighted_mse_wspd_rad(pred, target, mask, wmap, eps=1e-8):
    """
    Loss = mean(Wmse * Wwspd * Wrad) / mean(Wwspd * Wrad)
         = sum((pred-target)^2 * wmap) / sum(wmap)
    """
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if target.ndim == 3: target = target.unsqueeze(1)
    if mask.ndim == 3: mask = mask.unsqueeze(1)

    mse = (pred - target) ** 2
    num = (mse * wmap).sum()
    den = wmap.sum().clamp_min(eps)
    return num / den

import torch

def build_radius_map(H, W, device):
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    return torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)  # (H,W) en pixels


@torch.no_grad()
def compute_vmax_rmax_constants_from_loader(train_loader, device, max_batches=None):
    """
    Calcule une seule fois :
      VMAX_TRAIN = max over images (max pixel SAR normalisé)
      RMIN_TRAIN = min over images (rmax en pixels via profil radial)
    """
    vmax_list = []
    rmax_list = []

    r_map = None
    r_bins = None

    for b, (x, sar, mask, infos) in enumerate(train_loader):
        if max_batches is not None and b >= max_batches:
            break

        sar = sar.to(device)
        mask = mask.to(device)

        if sar.ndim == 3: sar = sar.unsqueeze(1)
        if mask.ndim == 3: mask = mask.unsqueeze(1)

        B, _, H, W = sar.shape

        # init r_map / r_bins une seule fois
        if r_map is None:
            r_map = build_radius_map(H, W, device)           # (H,W)
            r_max = int(min(H, W) // 2)
            r_bins = torch.arange(0, r_max, device=device)   # (R,)

        sar = torch.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- VMAX par image ----
        for i in range(B):
            valid = mask[i, 0] > 0.5
            if valid.any():
                vmax_list.append(sar[i, 0][valid].max())
            else:
                # si pas de pixels valides, on ignore
                continue

        # ---- RMAX par image via profil radial ----
        for i in range(B):
            t = sar[i, 0]
            valid = mask[i, 0] > 0.5
            if not valid.any():
                continue

            prof = torch.zeros((r_bins.numel(),), device=device, dtype=torch.float32)
            prev = 0.0
            for j, r0 in enumerate(r_bins):
                ring = (r_map >= r0) & (r_map < r0 + 1) & valid
                if ring.any():
                    prev = t[ring].mean()
                prof[j] = prev  # carry forward si ring vide

            rmax_list.append(r_bins[torch.argmax(prof)].float())

    if len(vmax_list) == 0 or len(rmax_list) == 0:
        raise RuntimeError("Impossible de calculer VMAX/RMAX : pas de pixels valides dans le train.")

    vmax_all = torch.stack(vmax_list)  # (N,)
    rmax_all = torch.stack(rmax_list)  # (N,)

    VMAX_TRAIN = float(vmax_all.max().item())
    RMIN_TRAIN = float(rmax_all.mean().item())

    return VMAX_TRAIN, RMIN_TRAIN

