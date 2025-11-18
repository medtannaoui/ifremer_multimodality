"""
Data preprocessing utilities for IR → SAR learning.

This module provides:
    - Loading IR/SAR tensors from pkl
    - Resampling IR to SAR spatial resolution
    - SAR missing-value handling (mask or fill)
    - Adding channel dimension (N → N,1,H,W)
    - Normalizing tensors

Author: Mohammed Amine Tannaoui
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import pickle as pkl
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from src.visualisation.utils_colormap import CMAP
from src.IR_to_SAR.distribution_data_visualisation import detect_sar_quadrants



# ============================================================
# ========================= DATA LOADING ======================
# ============================================================

def load_ir_sar_tensors(pkl_path: str = "/scale/user/mtannaou/alternance/src/IR_to_SAR/irwin_wind_tensors.pkl"):
    """
    Loads IR and SAR tensors from a pickle file.

    Returns:
        ir_tensor  (np.ndarray): shape (H_ir, W_ir)
        sar_tensor (np.ndarray): shape (H_sar, W_sar)
    """
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    # We assume the pickle contains 2 entries only
    ir_tensor  = data[list(data.keys())[0]]
    sar_tensor = data[list(data.keys())[1]]

    return ir_tensor, sar_tensor


# ============================================================
# ============== IR RESAMPLING TO SAR RESOLUTION =============
# ============================================================

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
    center_idx = np.array(ir_tensor).shape[0] // 2
    half_size  = max_radius_km // 2

    ir_crop = np.array(ir_tensor)[
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


# ============================================================
# =================== SAR MISSING VALUE HANDLING =============
# ============================================================

def process_sar_missing_values(
    sar_tensor: np.ndarray,
    mode: str = "mask",
    fill_value: float = -1.0
):
    """
    Handles missing values (NaN) in SAR tensor.

    Inputs:
        sar_tensor : (H, W)
        mode       : "mask" → return tensor + mask
                      "fill" → fill NaN with a constant
        fill_value : value used if mode="fill"

    Returns:
        If mode="mask": (sar_tensor, sar_mask)
        If mode="fill": (sar_filled, sar_mask)

        sar_mask is boolean array of shape (H, W)
    """

    sar_mask = ~np.isnan(sar_tensor)

    if mode == "mask":
        return sar_tensor, sar_mask

    elif mode == "fill":
        sar_filled = sar_tensor.copy()
        sar_filled[np.isnan(sar_filled)] = fill_value
        return sar_filled, sar_mask

    else:
        raise ValueError("mode must be 'mask' or 'fill'")


# ============================================================
# ======================== NORMALIZATION ======================
# ============================================================


def min_max_normalize_numpy(tensor):
    """
    Min-Max normalization of a NumPy tensor (N, H, W),
    mapping values to [0, 1] while preserving NaN values.
    
    Args:
        tensor (numpy.ndarray): Input tensor of shape (N, H, W)
    
    Returns:
        normalized_tensor (numpy.ndarray)
        min_val (float)
        max_val (float)
    """
    min_val = np.nanmin(tensor)
    max_val = np.nanmax(tensor)

    # Avoid division by zero if max == min
    if max_val - min_val == 0:
        return tensor.copy(), min_val, max_val

    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    return normalized_tensor, min_val, max_val



# ============================================================
# ======================== split 3 sets ======================
# ============================================================
from collections import Counter
def train_val_test_split(
    ir_array,
    sar_array,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    n_bins=3
):
    """
    Split IR→SAR dataset while preserving:
        - Brightness temperature distribution
        - Wind speed distribution
        - Quadrant spatial distribution

    Handles NaNs by temporary replacement with -1000 for stratification,
    but returns original tensors WITH true NaNs.
    """

    assert abs(train_size + val_size + test_size - 1) < 1e-6

    N = len(ir_array)

    # --- Create temporary clean versions (replace NaN by -1000) ---
    ir_clean = np.nan_to_num(ir_array, nan=-1000)
    sar_clean = np.nan_to_num(sar_array, nan=-1000)

    # --- 1️⃣ Compute median brightness & wind ---
    ir_median = np.median(ir_clean, axis=(1,2))
    sar_median = np.median(sar_clean, axis=(1,2))

    # --- 2️⃣ Encode into bins ---
    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    temp_bin = kb.fit_transform(ir_median.reshape(-1,1)).astype(int).flatten()
    wind_bin = kb.fit_transform(sar_median.reshape(-1,1)).astype(int).flatten()

    # --- 3️⃣ Quadrant detection using REAL NaN data ---
    quadrant_flags = []
    for i in range(N):
        q = detect_sar_quadrants(sar_array[i])  # Use original SAR with NaNs
        quadrant_flags.append("".join(map(str, q)))
    quadrant_flags = np.array(quadrant_flags)

    # --- 4️⃣ Create combined stratification label ---
    stratify_label = temp_bin.astype(str) + "_" + wind_bin.astype(str) + "_" + quadrant_flags

    # --- 5️⃣ Handle rare classes (<3 samples) ---
    counts = Counter(stratify_label)
    rare_classes = {k for k, v in counts.items() if v < 3}
    print(f"📉 {len(rare_classes)} rare stratification classes reassigned to 'OTHERS'")

    stratify_label_cleaned = np.array([
        lbl if lbl not in rare_classes else "OTHERS"
        for lbl in stratify_label
    ])

    # --- 6️⃣ Split train + temp (val+test) ---
    X = np.arange(N)
    X_train, X_temp, _, strat_temp = train_test_split(
        X,
        stratify_label_cleaned,
        stratify=stratify_label_cleaned,
        test_size=(1 - train_size),
        random_state=42
    )

    # --- 7️⃣ Split val/test ---
    val_rel_size = val_size / (val_size + test_size)

    # Re-check class frequencies in strat_temp
    counts_temp = Counter(strat_temp)
    rare_temp = {k for k, v in counts_temp.items() if v < 2}

    if rare_temp:
        print(f"🔁 Merging {len(rare_temp)} rare classes before val/test split...")
        strat_temp_cleaned = np.array([
            lbl if lbl not in rare_temp else "OTHERS"
            for lbl in strat_temp
        ])
    else:
        strat_temp_cleaned = strat_temp

    X_val, X_test = train_test_split(
        X_temp,
        stratify=strat_temp_cleaned,
        test_size=(1 - val_rel_size),
        random_state=42
    )
    print("🎯 Final Split Sizes:", len(X_train), len(X_val), len(X_test))

    # --- 8️⃣ Return ORIGINAL arrays (with true NaNs preserved) ---

    X_train = np.array([int(i) for i in X_train])
    X_val   = np.array([int(i) for i in X_val])
    X_test  = np.array([int(i) for i in X_test])

    # Debug (optional)
    print("Index types:", type(X_train), X_train.dtype)
    print("Sample values:", X_train[:10])
    ir_array = np.array(ir_array, dtype=float)
    sar_array = np.array(sar_array, dtype=float)
    return {
        "train": (ir_array[X_train], sar_array[X_train]),
        "val":   (ir_array[X_val],   sar_array[X_val]),
        # "test":  (ir_array[X_test],  sar_array[X_test]),
    }