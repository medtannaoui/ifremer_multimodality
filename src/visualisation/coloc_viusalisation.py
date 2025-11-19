"""
Visualization utilities for TCPrimed–SARGEO colocations.

This module loads a colocation PKL file that contains:
    - Microwave data (MW)
    - IRWIN from TCPrimed
    - SAR AEQD (wind speed)
    - IRWIN from SARGEO

It then generates a 4-panel visualization comparing all products on the same
AEQD / cyclone-centered coordinate grid.

Author: Mohammed Amine Tannaoui
"""

import os
import pickle as pkl
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from src.visualisation.utils_colormap import CMAP



# ============================================================
# ================ LOAD COLOCATION PKL FILE ==================
# ============================================================

def load_colocation_pkl(
    root_dir: str = "src/data_coloc_pkl",
    pattern: str = "2024",
    nth: int = 1
):
    """
    Searches and loads the N-th PKL file whose filename contains a pattern.

    Inputs:
        root_dir : directory containing colocation .pkl files
        pattern  : substring to match (ex: "2024")
        nth      : which matched file to load (1 = first match)

    Returns:
        data     : content of the pickle file (list of datasets)
        fullpath : path of the loaded file
    """

    matches = [f for f in os.listdir(root_dir) if pattern in f]

    if len(matches) == 0:
        raise FileNotFoundError(f"No PKL file in '{root_dir}' contains '{pattern}'.")

    matches.sort()
    if nth > len(matches):
        raise IndexError(f"Only {len(matches)} files found, cannot select nth={nth}.")

    fullpath = os.path.join(root_dir, matches[nth - 1])

    with open(fullpath, "rb") as f:
        data = pkl.load(f)

    return data, fullpath



# ============================================================
# ============ PLOT SINGLE COLOCATION (4-PANEL) ==============
# ============================================================

def plot_colocation(
    pattern: str = "2024",
    nth: int = 1,
    lim_km: int = 300,
    save: bool = False,
    save_dir: str = "src/visualisation/colocs_exemples_pics"
):
    """
    Visualizes TCPrimed–SARGEO colocated fields (MW, IRWIN, SAR, IR SARGEO).

    Inputs:
        pattern : substring to search in PKL filenames
        nth     : pick the nth file that matches pattern
        lim_km  : limit for x/y axes (centered, in km)
        save    : whether to save the figure
        save_dir: save directory if save=True
    """

    # ------------------------------
    # Load PKL file
    # ------------------------------
    data, filepath = load_colocation_pkl(pattern=pattern, nth=nth)

    cmap_ir  = CMAP.cira_ir()
    cmap_sar = CMAP.cmap_sar()

    # ------------------------------
    # Extract content
    # ------------------------------
    # data structure:
    # 0 → MW   : [x_mw, y_mw, TB_mw]
    # 1 → IRTC : [IRWIN, x_ir, y_ir]
    # 2 → SAR  : [x_sar, y_sar, wind_sar]
    # 3 → IRSG : [IR_sargeo, x_sg, y_sg, lat, lon]

    x_mw,  y_mw,  mw_field      = data[0]
    ir_tc, x_ir, y_ir           = data[1][:3]
    x_sar, y_sar, sar_field     = data[2]
    ir_sg, x_sg, y_sg, lat_c, lon_c = data[3]

    # Temperature conversions
    ir_tc_c  = ir_tc  - 273.15
    ir_sg_c  = ir_sg  - 273.15
    sar_kn   = sar_field * 1.94384   # m/s → knots

    # ------------------------------
    # Create figure layout
    # ------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # -- 1) Microwave -----------------------------------------
    ax = axes[0, 0]
    p = ax.pcolormesh(x_mw, y_mw, mw_field, shading="auto")
    ax.set_title("Microwave Brightness Temperature", fontweight="bold")
    ax.set_xlim(-lim_km, lim_km)
    ax.set_ylim(-lim_km, lim_km)
    fig.colorbar(p, ax=ax)

    # -- 2) IRWIN TCPrimed ------------------------------------
    ax = axes[0, 1]
    p = ax.pcolormesh(x_ir, y_ir, ir_tc_c, cmap=cmap_ir, shading="auto", vmin=-100, vmax=40)
    ax.set_title("IRWIN TCPrimed", fontweight="bold")
    ax.set_xlim(-lim_km, lim_km)
    ax.set_ylim(-lim_km, lim_km)
    fig.colorbar(p, ax=ax)

    # -- 3) SAR AEQD ------------------------------------------
    ax = axes[1, 0]
    p = ax.pcolormesh(x_sar, y_sar, sar_kn, cmap=cmap_sar, shading="auto", vmin=0, vmax=150)
    ax.set_title("SAR Wind Speed (AEQD)", fontweight="bold")
    ax.set_xlim(-lim_km, lim_km)
    ax.set_ylim(-lim_km, lim_km)
    fig.colorbar(p, ax=ax)

    # -- 4) IR SARGEO -----------------------------------------
    ax = axes[1, 1]
    p = ax.pcolormesh(x_sg, y_sg, ir_sg_c, cmap=cmap_ir, shading="auto", vmin=-100, vmax=40)
    ax.set_title("IRWIN SARGEO", fontweight="bold")
    ax.set_xlim(-lim_km, lim_km)
    ax.set_ylim(-lim_km, lim_km)
    fig.colorbar(p, ax=ax)

    # ------------------------------
    # Titles & Saving
    # ------------------------------
    fig.suptitle(os.path.basename(filepath), fontsize=16, fontweight="bold")

    for ax in axes.flat:
        ax.grid(False)

    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, os.path.basename(filepath) + f"_{lim_km}km.png")
        plt.savefig(out, dpi=150)
        print(f"Saved figure → {out}")

    plt.show()



# ============================================================
# =========================== MAIN ===========================
# ============================================================

if __name__ == "__main__":
    plot_colocation(pattern="2024", nth=2, lim_km=300)
