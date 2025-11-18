"""
Visualization utilities for IR → SAR data exploration.

This module provides:
    - Loading IRWIN and SAR tensors from SARGEO CSV
    - Histogram visualizations (brightness temperature & wind speed)
    - IR to SAR grid alignment
    - SAR to IR resolution downsampling
    - Scatter plots: IR vs SAR (pixel-to-pixel)
    - Distribution comparison for SAR missing pixels

Author: Mohammed Amine Tannaoui
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import zoom



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