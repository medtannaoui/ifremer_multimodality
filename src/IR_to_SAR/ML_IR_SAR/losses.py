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
    alpha=0.5,
    eps=1e-6,
    max_batches=None
):
    num_bins = bin_edges.numel() - 1
    counts = torch.zeros(num_bins, device=device, dtype=torch.float64)

    for b, batch in enumerate(train_loader):

        if max_batches is not None and b >= max_batches:
            break

        sar = batch[1].to(device)
        mask = batch[2].to(device)

        sar = torch.nan_to_num(
            sar,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        valid = mask > 0.5

        if valid.sum() == 0:
            continue

        v = sar[valid].float()

        idx = torch.bucketize(
            v,
            bin_edges,
            right=False
        ) - 1

        idx = idx.clamp(
            0,
            num_bins - 1
        )

        counts += torch.bincount(
            idx,
            minlength=num_bins
        ).to(torch.float64)

    probs = counts / counts.sum().clamp_min(1.0)

    weights = 1.0 / torch.pow(
        probs + eps,
        alpha
    )

    weights = weights / weights.mean().clamp_min(1e-12)

    return (
        weights.to(torch.float32),
        probs.to(torch.float32),
        counts
    )

@torch.no_grad()
def compute_bin_edges_quantiles(
    train_loader,
    device,
    num_bins=5,
    max_batches=None,
    eps=1e-6,
):

    vmin = None
    vmax = None

    for b, batch in enumerate(train_loader):

        if max_batches is not None and b >= max_batches:
            break

        sar = batch[1].to(device)
        mask = batch[2].to(device)

        sar = torch.nan_to_num(
            sar,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        valid = mask > 0.5

        if valid.sum() == 0:
            continue

        v = sar[valid].float()

        bmin = v.min()
        bmax = v.max()

        vmin = (
            bmin
            if vmin is None
            else torch.minimum(vmin, bmin)
        )

        vmax = (
            bmax
            if vmax is None
            else torch.maximum(vmax, bmax)
        )

    if vmin is None or vmax is None:
        raise RuntimeError(
            "No valid pixels found to compute min/max bin edges."
        )

    if torch.isclose(vmin, vmax):
        vmax = vmin + eps

    edges = torch.linspace(
        vmin,
        vmax,
        steps=num_bins + 1,
        device=device
    )

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


def temporal_neighbor_mse_loss(
    pred,
    weight_distance_1=0.5,
    weight_distance_2=0.25,
):
    pred = _ensure_4d(pred)

    _, number_of_channels, _, _ = pred.shape

    zero = pred.sum() * 0.0

    if number_of_channels <= 1:
        return zero

    loss = zero
    active_weight = 0.0

    # Canaux voisins : t et t+1
    if number_of_channels >= 2 and weight_distance_1 > 0:

        loss_distance_1 = F.mse_loss(
            pred[:, 1:, :, :],
            pred[:, :-1, :, :],
            reduction="mean",
        )

        loss = loss + weight_distance_1 * loss_distance_1
        active_weight += weight_distance_1

    # Canaux séparés de deux pas : t et t+2
    if number_of_channels >= 3 and weight_distance_2 > 0:

        loss_distance_2 = F.mse_loss(
            pred[:, 2:, :, :],
            pred[:, :-2, :, :],
            reduction="mean",
        )

        loss = loss + weight_distance_2 * loss_distance_2
        active_weight += weight_distance_2

    if active_weight > 0:
        loss = loss / active_weight

    return loss

def temporal_pixel_stability_loss(
    pred,
    spatial_mask=None,
    eps=1e-8,
):
    pred = _ensure_4d(pred)

    _, number_of_channels, _, _ = pred.shape

    zero = pred.sum() * 0.0

    if number_of_channels <= 1:
        return zero

    # Moyenne temporelle pour chaque pixel
    temporal_mean = pred.mean(
        dim=1,
        keepdim=True,
    )

    # Variance temporelle par pixel
    temporal_variance = (
        pred - temporal_mean
    ).pow(2).mean(
        dim=1,
        keepdim=True,
    )

    if spatial_mask is None:
        return temporal_variance.mean()

    spatial_mask = _ensure_4d(
        spatial_mask
    ).to(
        device=pred.device,
        dtype=pred.dtype,
    )

    if spatial_mask.shape[1] != 1:
        spatial_mask = spatial_mask.any(
            dim=1,
            keepdim=True,
        ).to(pred.dtype)

    if spatial_mask.shape != temporal_variance.shape:
        raise ValueError(
            "Le masque spatial doit être compatible avec "
            f"{temporal_variance.shape}, reçu {spatial_mask.shape}."
        )

    numerator = (
        temporal_variance * spatial_mask
    ).sum()

    denominator = spatial_mask.sum().clamp_min(eps)

    return numerator / denominator

def temporal_combined_sar_loss(
    sar_valid,
    pred_valid,
    mask,
    # Poids internes de la loss principale
    w_pix=1.0,
    w_grad=0.0,
    w_radial=0.0,
    r_bins=None,
    bin_edges=None,
    bin_weights=None,
    use_weighted_pix=False,
    # Poids des deux régularisations temporelles
    lambda_neighbors=1e-5,
    lambda_stability=1e-10,
    # Poids internes des distances temporelles
    neighbor_weight_1=0.5,
    neighbor_weight_2=0.25,
    use_spatial_mask_for_stability=False,
    eps=1e-8,
):
    sar_valid = _ensure_4d(sar_valid)
    pred_valid = _ensure_4d(pred_valid)
    mask = _ensure_4d(mask).to(
        device=pred_valid.device,
        dtype=pred_valid.dtype,
    )

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

    # 1. Terme principal : doit rester dominant
    (
        main_loss,
        l_pix,
        l_grad,
        l_radial,
    ) = combined_sar_loss(
        sar_valid=sar_valid,
        pred_valid=pred_valid,
        mask=mask,
        w_pix=w_pix,
        w_grad=w_grad,
        w_radial=w_radial,
        r_bins=r_bins,
        bin_edges=bin_edges,
        bin_weights=bin_weights,
        use_weighted_pix=use_weighted_pix,
        eps=eps,
    )

    zero = pred_valid.sum() * 0.0

    neighbor_loss = zero
    stability_loss = zero

    # 2. Cohérence temporelle locale
    if lambda_neighbors > 0:

        neighbor_loss = temporal_neighbor_mse_loss(
            pred=pred_valid,
            weight_distance_1=neighbor_weight_1,
            weight_distance_2=neighbor_weight_2,
        )

    # 3. Stabilité temporelle globale par pixel
    if lambda_stability > 0:
        spatial_mask = None
        if use_spatial_mask_for_stability:
            spatial_mask = mask.any(
                dim=1,
                keepdim=True,
            )
        stability_loss = temporal_pixel_stability_loss(
            pred=pred_valid,
            spatial_mask=spatial_mask,
            eps=eps,
        )

    # Important :
    # on ne divise PAS par la somme des lambdas.
    # La loss principale reste ainsi dominante.
    total_loss = (
        main_loss
        + lambda_neighbors * neighbor_loss
        + lambda_stability * stability_loss
    ) / (1+lambda_neighbors+lambda_stability)

    return (
        total_loss,
        l_pix,
        l_grad,
        l_radial
    )
