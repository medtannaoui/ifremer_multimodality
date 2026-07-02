import numpy as np
from matplotlib import pyplot as plt
import os




def _create_unique_dir(base_dir):
    i = 1
    while (base_dir / f"train_ir_sar_{i}").exists():
        i += 1
    return base_dir / f"train_ir_sar_{i - 1}"

def compute_vmax1d_rmax1d(sar_2d, resolution = 2):
    """
    Calcule Vmax1D et Rmax1D sur un champ SAR 2D.
        - Vmax1D : vitesse maximale du profil radial.
        - Rmax1D : rayon correspondant à Vmax1D.
    """
    H, W = sar_2d.shape
    cx, cy = W // 2, H // 2
    Y, X = np.indices((H, W))
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    Rmax = int(R.max())
    radii = np.arange(0, Rmax)
    vmax_profile = []
    for R0 in radii:
        mask = np.abs(R - R0) < 0.5
        vmax_profile.append(np.nanmax(sar_2d[mask]) if np.sum(mask) > 0 else np.nan)
    vmax_profile = np.array(vmax_profile)
    valid = ~np.isnan(vmax_profile)
    vmax_profile = vmax_profile[valid]
    radii = radii[valid]
    vmax1d = np.nanmax(vmax_profile)
    rmax1d = radii[np.nanargmax(vmax_profile)]
    return vmax1d, rmax1d * resolution  # km (résolution 2 km/pixel)



# -----------------------------------------------------------------
        # Fonctions de dé-normalisation (locales, réutilisées ci-dessous)
        # -----------------------------------------------------------------
def denorm(t, mean, std):
    return t * (std + 1e-10) + mean

def annular_denormalization(images_norm, stats, bin_size=1):
    N, H, W = images_norm.shape
    cx, cy = H // 2, W // 2
    y, x_idx = np.indices((H, W))
    radius = np.sqrt((y - cy) ** 2 + (x_idx - cx) ** 2)
    radial_bins = (radius // bin_size).astype(np.int32)
    mean = stats["mean"]
    std = stats["std"]
    images = images_norm.copy()
    for b in range(len(mean)):
        images[:, radial_bins == b] = images[:, radial_bins == b] * std[b] + mean[b]
    return images

def moment_to_sar(moment):
    assert moment.ndim == 3, "moment must be (N, H, W)"
    N, H, W = moment.shape
    y, x_idx = np.indices((H, W))
    cy, cx = H // 2, W // 2
    r = np.sqrt((x_idx - cx) ** 2 + (y - cy) ** 2)
    return moment / np.maximum(r, 1.0)[None, :, :]

def _split_bins_from_train(values_train, n_intervals=3):
    v = np.asarray(values_train)
    v = v[np.isfinite(v)]
    return np.linspace(v.min(), v.max(), n_intervals + 1) if v.size > 0 else None

def _compute_stats(err):
    bias = np.mean(err)
    std = np.std(err)
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    return bias, std, rmse, mae

def _plot_4panel_error_hist(errors, cat_values, bins, title_prefix, unit, save_path, xlim=None):
    """Histogramme d'erreurs en 4 panneaux : global + 3 catégories."""
    errors = np.asarray(errors)
    cat_values = np.asarray(cat_values)
    ok = np.isfinite(errors) & np.isfinite(cat_values)
    errors, cat_values = errors[ok], cat_values[ok]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    def draw(ax, err_sub, cat_sub, subtitle):
        err_sub, cat_sub = np.asarray(err_sub), np.asarray(cat_sub)
        m = np.isfinite(err_sub) & np.isfinite(cat_sub)
        err_sub, cat_sub = err_sub[m], cat_sub[m]
        if err_sub.size == 0:
            ax.set_title(subtitle + "\n(empty)")
            ax.grid(True, linestyle="--", alpha=0.4)
            return
        bias, std, rmse, mae = _compute_stats(err_sub)
        med_cat = np.median(cat_sub)
        norm_bias = bias / med_cat if med_cat != 0 else np.nan
        ax.hist(err_sub, bins=40)
        ax.set_title(subtitle)
        ax.set_xlabel(f"Error ({unit})")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.4)
        txt = (
            f"bias = {bias:.2f} {unit}\n"
            f"norm_bias = {norm_bias:.4f} (bias/median)\n"
            f"median(cat) = {med_cat:.2f}\n"
            f"stddev = {std:.2f} {unit}\n"
            f"rmse = {rmse:.2f} {unit}\n"
            f"mae = {mae:.2f} {unit}\n"
            f"n = {err_sub.size}"
        )
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        if xlim is not None:
            ax.set_xlim(xlim)

    draw(axes[0], errors, cat_values, f"{title_prefix}\nAll cases")
    for k in range(3):
        lo, hi = bins[k], bins[k + 1]
        sel = (cat_values >= lo) & (cat_values <= hi if k == 2 else cat_values < hi)
        draw(axes[k + 1], errors[sel], cat_values[sel],
                f"{title_prefix}\nCat{k + 1}: [{lo:.1f}, {hi:.1f}]")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
