# this script will be used to visualize tcprimed and sargeo colocs

from importlib import reload
from matplotlib import pyplot as plt 
import xarray as xr
import netCDF4 as nc
import pandas as pd
import pickle as pkl
import os
from src.visualisation.utils_colormap import CMAP


pkl_coloc_path = "src/data_coloc_pkl"



def get_pkl_coloc_file_from_str_iter(str_find = "2024", iter = 1):
    coloc_exemple_pkl_path = None
    cpt_24 = 0
    for pkl_file in os.listdir("src/data_coloc_pkl") : 
        if str_find in pkl_file:
            cpt_24 += 1 
            coloc_exemple_pkl_path = os.path.join("src/data_coloc_pkl",pkl_file)
            if cpt_24 == iter:
                break
    with open(coloc_exemple_pkl_path,"rb") as f:
        pkl_file = pkl.load(f)


    return pkl_file,coloc_exemple_pkl_path


def plot_coloc_sargeo_tcprimed(str_find="2024", iter= 1, save = False, lim = 300): 

    pkl_data, coloc_exemple_pkl_path = get_pkl_coloc_file_from_str_iter(str_find=str_find, iter= iter)

    cmap_ir = CMAP.cira_ir()
    cmap_sar = CMAP.cmap_sar()
    # --- Extract data ---
    # Microwave
    x_mw, y_mw, temp_mw = pkl_data[0]

    # IRWIN 2D + coords
    irwin_img, x_ir, y_ir = pkl_data[1][:3]

    # SAR AEQD
    x_sar, y_sar, wind_sar = pkl_data[2]

    # IR SARGEO + coords
    ir_sargeo, x_ir_sar, y_ir_sar, lat_c, lon_c = pkl_data[3]


    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --------------------
    # 1) MW (Microwave)
    # --------------------
    ax = axes[0, 0]
    img1 = ax.pcolormesh(x_mw, y_mw, temp_mw, shading="auto")
    ax.set_title("Microwave Brightness Temperature", fontsize=13, fontweight='bold')
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    fig.colorbar(img1, ax=ax)

    # --------------------
    # 2) IRWIN
    # --------------------
    ax = axes[0, 1]
    img2 = ax.pcolormesh(x_ir, y_ir, irwin_img - 273.15, cmap= cmap_ir, shading="auto", vmin=-100, vmax=40)
    ax.set_title("IRWIN TCPRimed", fontsize=13, fontweight='bold')
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    fig.colorbar(img2, ax=ax)

    # --------------------
    # 3) SAR AEQD
    # --------------------
    ax = axes[1, 0]
    img3 = ax.pcolormesh(x_sar, y_sar, wind_sar* 1.94384, cmap=cmap_sar, shading="auto", vmin=0, vmax=150)
    ax.set_title("SAR Wind Speed (AEQD Projection)", fontsize=13, fontweight='bold')
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    fig.colorbar(img3, ax=ax)

    # --------------------
    # 4) IR SARGEO
    # --------------------
    ax = axes[1, 1]
    img4 = ax.pcolormesh(x_ir_sar, y_ir_sar, ir_sargeo - 273.15, cmap= cmap_ir, shading="auto", vmin=-100, vmax=40)
    ax.set_title("IRWIN SARGEO", fontsize=13, fontweight='bold')
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    fig.colorbar(img4, ax=ax)

    # Style
    for ax in axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)


    fig.suptitle(coloc_exemple_pkl_path[19:-3], fontsize=18, fontweight='bold')
    if save : 
        os.makedirs("src/visualisation/colocs_exemples_pics",exist_ok=True)
        plt.savefig(f"src/visualisation/colocs_exemples_pics/{coloc_exemple_pkl_path[19:-3]}-{lim}km.png")
    plt.tight_layout()
    plt.show()


    
if __name__ == "__main__" : 
    plot_coloc_sargeo_tcprimed(str_find="2024", iter= 2,lim=300)


