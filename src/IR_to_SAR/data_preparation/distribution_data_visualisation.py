"""
Visualization utilities for IR → SAR data exploration.

This module provides:
    - Loading IRWIN and SAR tensors from SARGEO CSV
    - Histogram visualizations (brightness temperature & wind speed)
    - IR to SAR grid alignment
    - SAR to IR resolution downsampling
    - Scatter plots: IR vs SAR (pixel-to-pixel)
    - Distribution comparison for SAR missing pixels

"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from src_new.visualisation.utils_colormap import CMAP

cmap_ir = CMAP.cira_ir()
cmap_sar = CMAP.cmap_sar()



# ============================================================
# ========================= DATA LOADING ======================
# ============================================================

def load_ir_sar_values(
    sargeo_sar_csv_path: str = "/scale/user/mtannaou/alternance/excels/SARGEO_SAR_v1.csv",
    flatten: bool = False
):
    """
    Loads IR and SAR values for all colocated products listed in SARGEO_SAR CSV.

    Inputs:
        sargeo_sar_csv_path : path to SARGEO_SAR CSV
        flatten              : flatten each field, or keep original 2D maps

    Returns:
        If flatten=True:
            IR_flat, SAR_flat (1D arrays)
        Else:
            IR_list, SAR_list (list of 2D arrays)
    """

    df = pd.read_csv(sargeo_sar_csv_path)

    # Skip obsolete AEQD data
    df = df[~df["sar_xy"].str.contains("donnees_sar_changes", na=False)]

    IR_list = []
    SAR_list = []

    for _, row in df.iterrows():

        sar_path = row["sar_xy"]
        ir_name  = row["fichier"]
        cyclone  = row["cyclone"]

        # Build IR file path
        sargeo_root = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v01r02/cyclobs"
        ir_path = os.path.join(sargeo_root, cyclone, "IRWIN", ir_name)

        if not isinstance(sar_path, str) or not isinstance(ir_path, str):
            continue

        try:
            ds_sar = xr.open_dataset(sar_path)
            ds_ir  = xr.open_dataset(ir_path)

            # IR in Celsius
            ir  = (ds_ir["IRWIN"].sel(t_rel=0) - 273.15).values   # (H,W)

            # SAR in knots
            sar = (ds_sar["owiWindSpeed"].values * 1.94384)        # (H,W)

            if flatten:
                IR_list.append(ir.flatten())
                SAR_list.append(sar.flatten())
            else:
                IR_list.append(ir)
                SAR_list.append(sar)

        except Exception as err:
            print(f"⚠️ Error loading {sar_path}: {err}")
            continue

    if flatten:
        return np.concatenate(IR_list), np.concatenate(SAR_list)

    return IR_list, SAR_list


# ============================================================
# =================== HISTOGRAM VISUALIZATION ================
# ============================================================

def plot_ir_hist(values, ax=None, title=""):

    if ax is None : 
        
        ax = plt.gca()
    """Plots IR brightness temperature distribution (°C)."""
    
    ax.hist(values, bins=200, color="gray")
    ax.set_xlabel("IR Brightness Temperature (°C)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of IR Brightness Temperature "+ str(title))
    ax.grid(alpha=0.3)
        


def plot_sar_hist(values,ax =None, title=""):

    """Plots SAR wind speed distribution (knots)."""
    if ax is None : 
        plt.figure(figsize=(7, 5))
        ax = plt.gca()
    ax.hist(values, bins=200, color="steelblue")
    ax.set_xlabel("SAR Wind Speed (kt)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of SAR Wind Speed "+ str(title))
    ax.grid(alpha=0.3)
   


# ============================================================
# ===================== SCATTER IR VS SAR =====================
# ============================================================

from src.visualisation.utils_colormap import CMAP
def resize_ir_to_sar(
    ir_tensor: np.ndarray,
    max_radius_km: int = 300,
    sar_resolution_km: float = 0.5,
    plot: bool = False
):
    """
    Resamples IR image to SAR AEQD grid resolution.

    Inputs:
        ir_tensor          : (H, W) original IR image
        max_radius_km      : radius in km to crop (final image spans [-R,+R])
        sar_resolution_km  : desired SAR resolution (ex: 0.5 km)
        plot               : show before/after comparison

    Returns:
        ir_resized    : (H_sar, W_sar) IR resized to match SAR resolution
    """

    cmap = CMAP.cira_ir()

    # Center and crop the IR image to match SAR diameter (600 km)
    center_idx = ir_tensor.shape[0] // 2
    half_size  = max_radius_km // 2

    ir_crop = ir_tensor[
        center_idx - half_size : center_idx + half_size,
        center_idx - half_size : center_idx + half_size
    ]  # shape = (300,300)

    # Compute scaling factor: original ~ 2 km/pixel → target sar_resolution (km/pixel)
    zoom_factor = 2.0 / sar_resolution_km
    ir_resized = zoom(ir_crop, zoom=zoom_factor, order=1)   # bilinear interpolation

    # Optional visualization
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Original crop
        ax[0].set_title("Original IR (cropped)")
        p0 = ax[0].imshow(ir_crop, cmap=cmap, origin="lower")
        plt.colorbar(p0, ax=ax[0])

        # Resized IR
        ax[1].set_title(f"IR resampled: {ir_resized.shape[0]}×{ir_resized.shape[1]} (resolution={sar_resolution_km} km)")
        p1 = ax[1].imshow(ir_resized, cmap=cmap, origin="lower")
        plt.colorbar(p1, ax=ax[1])

        plt.tight_layout()
        plt.show()

    return ir_resized

def scatter_ir_vs_sar(ir_tensor, sar_tensor, max_points=300000):
    """
    Pixel-to-pixel scatter plot IR (downsampled) vs SAR (downsampled).
    Assumes:
        ir_tensor  : (N,1001,1001)
        sar_tensor : (N,601,601)
    """
                      # 150*2km = 300km


    # 2. Downsample SAR (601→300)
    ir_resampled = resize_ir_to_sar(ir_tensor,max_radius_km=300,sar_resolution_km=1,plot=False)

    # 3. Flatten
    ir_flat  = ir_resampled.flatten()
    sar_flat = sar_tensor.flatten()

    # 4. Remove NaNs
    mask = np.isfinite(ir_flat) & np.isfinite(sar_flat)
    ir_valid  = ir_flat[mask]
    sar_valid = sar_flat[mask]

    # Optional subsampling for readability
    if len(ir_valid) > max_points:
        idx = np.random.choice(len(ir_valid), max_points, replace=False)
        ir_valid  = ir_valid[idx]
        sar_valid = sar_valid[idx]

    # 5. Plot
    plt.figure(figsize=(7, 7))
    plt.scatter(ir_valid, sar_valid, s=2, alpha=0.1)
    plt.xlabel("IR (°C) — downsampled to SAR resolution")
    plt.ylabel("SAR Wind Speed (kt)")
    plt.title("IR vs SAR — Pixel-to-Pixel Scatter Plot")
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================
# ============= IR DISTRIBUTION FOR SAR MISSING VALUES =======
# ============================================================

def compare_ir_distributions_sar_nan(ir_tensor, sar_tensor):
    """
    Compares IR distribution for pixels where SAR is valid vs SAR is NaN.

    Useful to analyse missing-value patterns.
    """

    # 1. Crop IR
    
    # 2. Downsample SAR
    ir_resampled = resize_ir_to_sar(ir_tensor,max_radius_km=300,sar_resolution_km=1,plot=False)

    # 3. Flatten
    ir_flat  = ir_resampled.flatten()
    sar_flat = sar_tensor.flatten()

    # 4. Masks
    mask_nan   = np.isnan(sar_flat)
    mask_valid = np.isfinite(sar_flat)

    ir_nan     = ir_flat[mask_nan]
    ir_valid   = ir_flat[mask_valid]

    # 5. Plot
    plt.figure(figsize=(8, 5))
    plt.hist(ir_valid, bins=100, alpha=0.6, color="green", label="SAR Valid")
    plt.hist(ir_nan,   bins=100, alpha=0.6, color="red",   label="SAR NaN")
    plt.xlabel("IR Temperature (°C)")
    plt.ylabel("Count")
    plt.title("IR Distributions for SAR Valid vs SAR Missing")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================
# ============= DETECT SAR QUADRANTS =========================
# ============================================================

def detect_sar_quadrants(sar_tensor):
    """
    Detect which quadrants contain valid (non-NaN) SAR data.
    Assumes domain is centered and shape is (H,W).
    
    Returns a numpy array [NW, NE, SW, SE], each 0 or 1.
    """
    
    H, W = sar_tensor.shape
    center = H // 2
    
    # Create a mask of valid pixels (True where data exists)
    valid = ~np.isnan(sar_tensor)

    quadrants = np.zeros(4, dtype=int)

    # NW: top-left
    quadrants[0] = np.any(valid[0:center, 0:center])

    # NE: top-right
    quadrants[1] = np.any(valid[0:center, center:W])

    # SW: bottom-left
    quadrants[2] = np.any(valid[center:H, 0:center])

    # SE: bottom-right
    quadrants[3] = np.any(valid[center:H, center:W])

    return quadrants




def plot_quadrant_distribution(sar_tensors, ax=None, title = ""):
    """
    Analyzes and visualizes the distribution of valid SAR coverage
    across the four quadrants (NW, NE, SW, SE) in the dataset.
    """

    if ax is None:
        ax = plt.gca()

    quadrant_list = []
    for tensor in sar_tensors:
        q = detect_sar_quadrants(tensor)  # returns [NW, NE, SW, SE]
        quadrant_list.append(q)

    quadrant_array = np.array(quadrant_list)
    quadrant_counts = quadrant_array.sum(axis=0)
    total_samples = len(sar_tensors)
    quadrant_percentages = (quadrant_counts / total_samples) * 100

    labels = ["NW", "NE", "SW", "SE"]

    # --- Print summary in console ---
    print("\n📊 Quadrant Coverage Distribution:")
    for i, label in enumerate(labels):
        print(f"  {label}: {quadrant_counts[i]} samples ({quadrant_percentages[i]:.1f}%)")

    # --- Bar Plot directly on ax ---
    bars = ax.bar(labels, quadrant_percentages)

    ax.set_title("SAR Coverage Distribution per Quadrant"+ str(title), fontsize=12)
    ax.set_ylabel("Presence Percentage (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Add text on bars
    for bar, pct in zip(bars, quadrant_percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            pct + 1,  # place text slightly above bar
            f"{pct:.1f}%",
            ha='center',
            va='bottom',
            fontsize=10
        )


def plot_sar(tensor, ax=None):
    if ax is None:
        ax = plt.gca()
    
    x_sar,y_sar = np.linspace(-300,300,tensor.shape[0]) , np.linspace(-300,300,tensor.shape[1])
    ax.pcolormesh(x_sar, y_sar, tensor, cmap=cmap_sar)
    ax.set_aspect('equal')


def plot_ir(tensor, ax=None):
    if ax is None : 
        ax=plt.gca()
    
    x_ir, y_ir = np.linspace(-300,300,tensor.shape[0]) , np.linspace(-300,300,tensor.shape[1])
    ax.pcolormesh(x_ir, y_ir , tensor, cmap=cmap_ir)
    ax.set_aspect('equal')


def vmax_compare(true_sars, predict_sars, output_dir, set, epoch, plot=False):
    if set == "train":
        return None

    # ✅ Vectorisation complète (beaucoup plus rapide)
    vmax1 = []
    vmax2 = []
    for sar1, sar2 in zip(true_sars, predict_sars):
        vmax1.append(np.nanmax(sar1))
        vmax2.append(np.nanmax(sar2))
    vmax1, vmax2 = np.array(vmax1), np.array(vmax2)

    # ✅ Statistiques calculées UNE seule fois
    vmin = min(np.nanmin(vmax1),  np.nanmin(vmax2))
    vmax = max(np.nanmax(vmax1), np.nanmax(vmax2))

    mean_true  = np.nanmean(vmax1)
    mean_pred  = np.nanmean(vmax2)
    max_true   = np.nanmax(vmax1)
    max_pred   = np.nanmax(vmax2)

    plt.figure(figsize=(5, 5))
    plt.scatter(vmax1, vmax2, alpha=0.5)
    plt.plot([vmin, vmax], [vmin, vmax], "r--", lw=2)

    plt.xlabel("Vmax True (knots)", fontsize=13)
    plt.ylabel("Vmax Predicted (knots)", fontsize=13)
    plt.title(f"Vmax Comparison — Epoch {epoch+1} ({set})", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)

    textstr = (
        f"Mean True  : {mean_true:.2f}\n"
        f"Mean Pred  : {mean_pred:.2f}\n"
        f"Max True   : {max_true:.2f}\n"
        f"Max Pred   : {max_pred:.2f}"
    )

    plt.text(
        0.05, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    if not plot:

        out_path = os.path.join(output_dir, f"Vmax_2D_comparison_{set}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()  
    else :
        plt.show()
    


def compare_sar_distribution(sar_knots, pred_knots, output_dir, set, epoch):
        if set == "train":
            return None
        plt.figure(figsize=(8, 6))
        plt.hist(sar_knots, bins=60, alpha=0.5, density=True, label="True SAR", color="blue")
        plt.hist(pred_knots, bins=60, alpha=0.5, density=True, label="Predicted SAR", color="orange")
        plt.legend()
        plt.xlabel("Wind Speed (knots)")
        plt.ylabel("Density")
        plt.title(f"Global Wind Speed Distribution —_{set}")
        out_path = os.path.join(output_dir, f"wind_distribution_{set}.png")
        plt.savefig(out_path, dpi=150)
        plt.close('all')

        print(f"📈 Saved global distribution")

def compare_radial_vmax(sars_true, sars_predict, output_dir, set, epoch,  center=None, dr=1, plot=False):
    if set =="train":
        return None
    
    assert sars_true.shape == sars_predict.shape, "sars1 and sars2 must have the same shape"

    N, H, W = sars_true.shape

    # --- Center
    if center is None:
        yc, xc = H // 2, W // 2
    else:
        yc, xc = center

    # --- Radial distance map
    Y, X = np.indices((H, W))
    R = np.sqrt((X - xc)**2 + (Y - yc)**2)

    Rmax = int(R.max())
    r_bins = np.arange(0, Rmax + dr, dr)

    vmax_r_1 = []
    vmax_r_2 = []
    r_centers = []

    # --- Radial vmax for each shell
    for r0 in r_bins[:-1]:
        mask = (R >= r0) & (R < r0 + dr)

        if np.sum(mask) == 0:
            continue

        vmax1 = np.nanmean(np.nanmax(sars_true[:, mask], axis=1))
        vmax2 = np.nanmean(np.nanmax(sars_predict[:, mask], axis=1))

        vmax_r_1.append(vmax1)
        vmax_r_2.append(vmax2)
        r_centers.append(r0 + dr / 2)

    vmax_r_1 = np.array(vmax_r_1)
    vmax_r_2 = np.array(vmax_r_2)

    # --- Error between the two curves
    error = np.abs(vmax_r_1 - vmax_r_2)

    # --- Plot
    plt.figure(figsize=(10, 6))

    plt.plot(r_centers, vmax_r_1, color="green", linewidth=2, label="SAR1 Radial Vmax")
    plt.plot(r_centers, vmax_r_2, color="blue", linewidth=2, label="SAR2 Radial Vmax")
    plt.plot(r_centers, error, color="red", linestyle="--", linewidth=2, label="Absolute Error")

    plt.xlabel("Radius R (pixels)", fontsize=14)
    plt.ylabel("Vmax Mean (knots)", fontsize=14)
    plt.title(f"Radial Distribution of Vmax and Error", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=12)

    plt.tight_layout()
    if plot :
        plt.show()
    else : 
        out_path = os.path.join(output_dir, f"radial_vmax_{set}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


def compute_mae_metric(sar_true, sar_pred, output_dir, set, epoch, plot=False):
    if set=="train":
        return None
    
    mae_global = np.nanmean(np.abs(sar_pred - sar_true))


    # =========================
    mae_map = np.nanmean(np.abs(sar_pred - sar_true), axis=0)  # (H, W)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(mae_map, cmap="inferno")
    plt.colorbar(im, label="MAE (knots)")
    plt.title(f"Mean Absolute Error Map — {set} (Epoch {epoch+1})")
    plt.axis("off")
    if not plot : 
        save_path = os.path.join(output_dir, f"mae_map_{set}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

        
    else : 
        plt.show()

    


def rmax_compare(true_sars, predict_sars, output_dir, set, epoch, plot=False):
    """
    true_sars, predict_sars : numpy arrays of shape (N, H, W)
    RMW computed in PIXELS assuming the cyclone center is at the image center.
    """

    if set == "train":
        return None

    N, H, W = true_sars.shape
    x_c, y_c = W // 2, H // 2

    Y, X = np.indices((H, W))
    dist_map = np.sqrt((X - x_c)**2 + (Y - y_c)**2)

    rmw_true = []
    rmw_pred = []

    for sar_t, sar_p in zip(true_sars, predict_sars):

        # --- True RMW ---
        vmax_t = np.nanmax(sar_t)
        mask_t = sar_t == vmax_t
        rmw_t  = np.nanmean(dist_map[mask_t])

        # --- Pred RMW ---
        vmax_p = np.nanmax(sar_p)
        mask_p = sar_p == vmax_p
        rmw_p  = np.nanmean(dist_map[mask_p])

        rmw_true.append(rmw_t)
        rmw_pred.append(rmw_p)

    rmw_true = np.array(rmw_true)
    rmw_pred = np.array(rmw_pred)

    rmin = min(np.nanmin(rmw_true), np.nanmin(rmw_pred))
    rmax = max(np.nanmax(rmw_true), np.nanmax(rmw_pred))

    mean_true = np.nanmean(rmw_true)
    mean_pred = np.nanmean(rmw_pred)

    max_true = np.nanmax(rmw_true)
    max_pred = np.nanmax(rmw_pred)

    plt.figure(figsize=(5, 5))
    plt.scatter(rmw_true, rmw_pred, alpha=0.5)
    plt.plot([rmin, rmax], [rmin, rmax], "r--", lw=2)

    plt.xlabel("Rmax True (pixels)", fontsize=13)
    plt.ylabel("Rmax Predicted (pixels)", fontsize=13)
    plt.title(f"Rmax Comparison —({set})", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)

    textstr = (
        f"Mean True  : {mean_true:.1f}\n"
        f"Mean Pred  : {mean_pred:.1f}\n"
        f"Max True  : {max_true:.1f}\n"
        f"Max Pred  : {max_pred:.1f}\n"
    )

    plt.text(
        0.05, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    if not plot:
        out_path = os.path.join(output_dir, f"Rmaxcomparison_{set}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()

    

    

