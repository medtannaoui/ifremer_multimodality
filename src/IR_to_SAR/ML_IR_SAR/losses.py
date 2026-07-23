# this script will be sued for create losses sued in train
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional import structural_similarity_index_measure as ssim

def _ensure_4d(tensor):
    """
    Convertit les tenseurs en (B, C, H, W).

    Formats acceptés :
        (B, H, W)    -> (B, 1, H, W)
        (B, C, H, W) -> inchangé
    """
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)

    if tensor.ndim != 4:
        raise ValueError(
            f"Tensor attendu avec 3 ou 4 dimensions, reçu : {tensor.shape}"
        )

    return tensor


# ============================================================
# 1. LOSS PIXEL
# ============================================================

def pix2pix_l1_loss(
    sar_valid,
    pred_valid,
    mask,
    eps=1e-8,
):
    """
    L1 calculée uniquement sur les pixels valides.

    Fonctionne avec :
        (B, 1, H, W)
        (B, 12, H, W)

    Les canaux sans SAR ont un masque nul partout et ne
    participent donc pas à la loss.
    """

    sar_valid = _ensure_4d(sar_valid)
    pred_valid = _ensure_4d(pred_valid)
    mask = _ensure_4d(mask).to(pred_valid.dtype)

    if pred_valid.shape != sar_valid.shape:
        raise ValueError(
            f"pred et SAR ont des formes différentes : "
            f"{pred_valid.shape} != {sar_valid.shape}"
        )

    if mask.shape != sar_valid.shape:
        raise ValueError(
            f"mask et SAR ont des formes différentes : "
            f"{mask.shape} != {sar_valid.shape}"
        )

    absolute_error = torch.abs(pred_valid - sar_valid)

    numerator = (absolute_error * mask).sum()
    denominator = mask.sum().clamp_min(eps)

    return numerator / denominator


# ============================================================
# 2. LOSS GRADIENT
# ============================================================

def gradient_l1_loss(
    sar_valid,
    pred_valid,
    mask,
    eps=1e-8,
):
    """
    L1 entre les gradients SAR et prédiction.

    La loss est calculée uniquement lorsque les deux pixels
    voisins utilisés pour calculer un gradient sont valides.

    Fonctionne avec :
        (B, 1, H, W)
        (B, 12, H, W)
    """

    sar_valid = _ensure_4d(sar_valid)
    pred_valid = _ensure_4d(pred_valid)
    mask = _ensure_4d(mask).to(pred_valid.dtype)

    if pred_valid.shape != sar_valid.shape:
        raise ValueError(
            f"pred et SAR ont des formes différentes : "
            f"{pred_valid.shape} != {sar_valid.shape}"
        )

    if mask.shape != sar_valid.shape:
        raise ValueError(
            f"mask et SAR ont des formes différentes : "
            f"{mask.shape} != {sar_valid.shape}"
        )

    # Gradient horizontal : différence entre deux colonnes voisines
    grad_x_sar = sar_valid[..., :, 1:] - sar_valid[..., :, :-1]
    grad_x_pred = pred_valid[..., :, 1:] - pred_valid[..., :, :-1]

    # Les deux pixels doivent être valides
    mask_x = mask[..., :, 1:] * mask[..., :, :-1]

    # Gradient vertical : différence entre deux lignes voisines
    grad_y_sar = sar_valid[..., 1:, :] - sar_valid[..., :-1, :]
    grad_y_pred = pred_valid[..., 1:, :] - pred_valid[..., :-1, :]

    # Les deux pixels doivent être valides
    mask_y = mask[..., 1:, :] * mask[..., :-1, :]

    error_x = torch.abs(grad_x_pred - grad_x_sar) * mask_x
    error_y = torch.abs(grad_y_pred - grad_y_sar) * mask_y

    loss_x = error_x.sum() / mask_x.sum().clamp_min(eps)
    loss_y = error_y.sum() / mask_y.sum().clamp_min(eps)

    return loss_x + loss_y


# ============================================================
# 3. LOSS RADIALE VMAX
# ============================================================

def radial_vmax_l1_loss(
    pred_valid,
    sar_valid,
    mask,
    r_bins=None,
    eps=1e-8,
):
    """
    Compare le maximum SAR et le maximum prédit pour chaque
    rayon autour du centre.

    Fonctionne avec :
        (B, 1, H, W)
        (B, 12, H, W)

    Dans le cas multicanal, seuls les couples (batch, canal)
    contenant au moins un pixel SAR valide sont sélectionnés.
    """

    sar_valid = _ensure_4d(sar_valid)
    pred_valid = _ensure_4d(pred_valid)
    mask = _ensure_4d(mask).to(pred_valid.dtype)

    if pred_valid.shape != sar_valid.shape:
        raise ValueError(
            f"pred et SAR ont des formes différentes : "
            f"{pred_valid.shape} != {sar_valid.shape}"
        )

    if mask.shape != sar_valid.shape:
        raise ValueError(
            f"mask et SAR ont des formes différentes : "
            f"{mask.shape} != {sar_valid.shape}"
        )

    B, C, H, W = sar_valid.shape

    # Pour chaque couple (batch, canal), indique si du SAR existe
    valid_channels = mask.flatten(start_dim=2).sum(dim=2) > 0
    # Forme : (B, C)

    # Aucun canal SAR valide dans ce batch
    if not valid_channels.any():
        return pred_valid.sum() * 0.0

    # Sélectionne uniquement les canaux contenant du SAR
    # (N_valid, H, W)
    selected_sar = sar_valid[valid_channels]
    selected_pred = pred_valid[valid_channels]
    selected_mask = mask[valid_channels]

    # Retour au format monocanal
    # (N_valid, 1, H, W)
    selected_sar = selected_sar.unsqueeze(1)
    selected_pred = selected_pred.unsqueeze(1)
    selected_mask = selected_mask.unsqueeze(1)

    N = selected_sar.shape[0]
    device = selected_sar.device
    dtype = selected_sar.dtype

    # Centre spatial
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )

    radius = torch.sqrt(
        (xx - cx) ** 2 +
        (yy - cy) ** 2
    )

    radius_indices = torch.floor(radius).long()

    if r_bins is None:
        number_of_radii = int(min(H, W) // 2)
    else:
        number_of_radii = int(len(r_bins))

    number_of_radii = max(number_of_radii, 1)

    radius_indices = torch.clamp(
        radius_indices,
        min=0,
        max=number_of_radii - 1,
    )

    # (N, 1, H*W)
    radius_indices = radius_indices.view(1, 1, -1).expand(
        N,
        1,
        H * W,
    )

    flat_mask = selected_mask.reshape(N, 1, H * W) > 0
    flat_sar = selected_sar.reshape(N, 1, H * W)
    flat_pred = selected_pred.reshape(N, 1, H * W)

    # Les pixels invalides ne doivent jamais devenir des maxima
    negative_large_value = torch.finfo(dtype).min

    flat_sar = flat_sar.masked_fill(
        ~flat_mask,
        negative_large_value,
    )

    flat_pred = flat_pred.masked_fill(
        ~flat_mask,
        negative_large_value,
    )

    radial_max_sar = torch.full(
        (N, 1, number_of_radii),
        negative_large_value,
        device=device,
        dtype=dtype,
    )

    radial_max_pred = torch.full(
        (N, 1, number_of_radii),
        negative_large_value,
        device=device,
        dtype=dtype,
    )

    radial_counts = torch.zeros(
        (N, 1, number_of_radii),
        device=device,
        dtype=dtype,
    )

    radial_max_sar.scatter_reduce_(
        dim=2,
        index=radius_indices,
        src=flat_sar,
        reduce="amax",
        include_self=True,
    )

    radial_max_pred.scatter_reduce_(
        dim=2,
        index=radius_indices,
        src=flat_pred,
        reduce="amax",
        include_self=True,
    )

    radial_counts.scatter_add_(
        dim=2,
        index=radius_indices,
        src=flat_mask.to(dtype),
    )

    # Un rayon participe uniquement s'il contient des pixels valides
    valid_radii = radial_counts > 0

    if not valid_radii.any():
        return pred_valid.sum() * 0.0

    radial_error = torch.abs(
        radial_max_pred - radial_max_sar
    )

    radial_error = radial_error * valid_radii.to(dtype)

    return (
        radial_error.sum()
        / valid_radii.sum().to(dtype).clamp_min(eps)
    )


# ============================================================
# 4. LOSS COMBINÉE
# ============================================================

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
    eps=1e-8,
):
    """
    Loss complète compatible avec :

        SAR monocanale :
            sar, pred, mask = (B, 1, H, W)

        SAR temporelle :
            sar, pred, mask = (B, 12, H, W)

    Les canaux sans SAR, dont le masque est nul partout,
    ne contribuent à aucune composante de la loss.
    """

    sar_valid = _ensure_4d(sar_valid)
    pred_valid = _ensure_4d(pred_valid)
    mask = _ensure_4d(mask).to(pred_valid.dtype)

    if pred_valid.shape != sar_valid.shape:
        raise ValueError(
            f"pred et SAR ont des formes différentes : "
            f"{pred_valid.shape} != {sar_valid.shape}"
        )

    if mask.shape != sar_valid.shape:
        raise ValueError(
            f"mask et SAR ont des formes différentes : "
            f"{mask.shape} != {sar_valid.shape}"
        )

    # Valeurs nulles connectées au graphe
    zero = pred_valid.sum() * 0.0

    l_pix = zero
    l_grad = zero
    l_radial = zero

    # Aucun SAR valide dans tout le batch
    if mask.sum() <= 0:
        return (
            zero,
            l_pix.detach().cpu(),
            l_grad.detach().cpu(),
            l_radial.detach().cpu(),
        )

    total_loss = zero
    active_weight_sum = 0.0

    # --------------------------------------------------------
    # Loss pixel
    # --------------------------------------------------------
    if w_pix > 0:

        if use_weighted_pix:
            if bin_edges is None or bin_weights is None:
                raise ValueError(
                    "bin_edges et bin_weights sont nécessaires "
                    "lorsque use_weighted_pix=True."
                )

            wmap = make_weight_map(
                sar_valid,
                mask,
                bin_edges,
                bin_weights,
            )

            weighted_mask = mask * wmap

            absolute_error = torch.abs(
                pred_valid - sar_valid
            )

            l_pix = (
                (absolute_error * weighted_mask).sum()
                / weighted_mask.sum().clamp_min(eps)
            )

        else:
            l_pix = pix2pix_l1_loss(
                sar_valid=sar_valid,
                pred_valid=pred_valid,
                mask=mask,
                eps=eps,
            )

        total_loss = total_loss + w_pix * l_pix
        active_weight_sum += w_pix

    # --------------------------------------------------------
    # Loss gradient
    # --------------------------------------------------------
    if w_grad > 0:

        l_grad = gradient_l1_loss(
            sar_valid=sar_valid,
            pred_valid=pred_valid,
            mask=mask,
            eps=eps,
        )

        total_loss = total_loss + w_grad * l_grad
        active_weight_sum += w_grad

    # --------------------------------------------------------
    # Loss radiale
    # --------------------------------------------------------
    if w_radial > 0:

        l_radial = radial_vmax_l1_loss(
            pred_valid=pred_valid,
            sar_valid=sar_valid,
            mask=mask,
            r_bins=r_bins,
            eps=eps,
        )

        total_loss = total_loss + w_radial * l_radial
        active_weight_sum += w_radial

    if active_weight_sum > 0:
        total_loss = total_loss / active_weight_sum

    return (
        total_loss,
        l_pix.detach().cpu(),
        l_grad.detach().cpu(),
        l_radial.detach().cpu(),
    )


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


def make_weight_map(
    target,
    mask,
    bin_edges,
    bin_weights,
):
    if target.ndim == 3:
        target = target.unsqueeze(1)

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    if target.shape != mask.shape:
        raise ValueError(
            f"target et mask doivent avoir la même forme : "
            f"{target.shape} != {mask.shape}"
        )

    device = target.device
    dtype = target.dtype

    bin_edges = bin_edges.to(device=device, dtype=dtype)
    bin_weights = bin_weights.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)

    idx = torch.bucketize(
        target.contiguous(),
        bin_edges,
        right=False,
    ) - 1

    idx = idx.clamp(
        min=0,
        max=bin_weights.numel() - 1,
    )

    wmap = bin_weights[idx]

    # Les pixels et canaux sans SAR ont un poids nul
    wmap = wmap * mask

    return wmap


def weighted_l1(
    pred,
    target,
    mask,
    wmap,
    eps=1e-8,
):
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)

    if target.ndim == 3:
        target = target.unsqueeze(1)

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    if wmap.ndim == 3:
        wmap = wmap.unsqueeze(1)

    if pred.shape != target.shape:
        raise ValueError(
            f"pred et target ont des formes différentes : "
            f"{pred.shape} != {target.shape}"
        )

    if mask.shape != target.shape:
        raise ValueError(
            f"mask et target ont des formes différentes : "
            f"{mask.shape} != {target.shape}"
        )

    if wmap.shape != target.shape:
        raise ValueError(
            f"wmap et target ont des formes différentes : "
            f"{wmap.shape} != {target.shape}"
        )

    diff = torch.abs(pred - target)

    numerator = (diff * wmap).sum()
    denominator = wmap.sum().clamp_min(eps)

    return numerator / denominator


