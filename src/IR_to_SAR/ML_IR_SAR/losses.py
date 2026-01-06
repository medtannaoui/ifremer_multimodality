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
    r_bins=None
):

    loss = 0.0
    loss_dict = {}
    l_radial, l_grad, l_pix = 0.0, 0.0, 0.0

    sum = w_pix+w_grad+w_radial
    if w_pix > 0:
        l_pix = pix2pix_l1_loss(sar_valid, pred_valid, mask)
        loss += w_pix * l_pix
        loss_dict["loss_pix"] = l_pix.detach()

    if w_grad > 0:
        l_grad = gradient_l1_loss(sar_valid, pred_valid, mask)
        loss += w_grad * l_grad
        loss_dict["loss_grad"] = l_grad.detach()

    if w_radial > 0:
        l_radial = radial_vmax_l1_loss(
            sar_valid,
            pred_valid,
            mask,
            r_bins=r_bins
        )
        loss += w_radial * l_radial
        loss_dict["loss_radial"] = l_radial.detach()
    loss /= sum
    loss_dict["loss_total"] = loss.detach()

    return loss, l_pix.detach(),l_grad.detach(),l_radial.detach()

       
