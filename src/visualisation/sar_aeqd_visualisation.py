# this script is used to visualize the sar aeqd with the IR data.

from importlib import reload

import os
import pandas as pd
import re
import numpy as np
# import GEO_to_XYcenter as xysar
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import xarray as xr
from importlib import reload
from src.visualisation.utils_colormap import CMAP

#get inter path 
def get_inter_path(path):
    inter_path = None

    path = os.path.basename(path)   #take the basenime directory

    if "_ll" in path : 
        return path.split("_ll")[0]
    else : 
        return path.split("_aeqd")[0]



def get_geo_path(path_sar_aeqd):
    sargeo_path = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v01r02/cyclobs"
    sargeo_df = pd.read_csv("/scale/user/mtannaou/alternance/excels/SARGEO_SAR.csv")

    inter_path = get_inter_path(path_sar_aeqd)
    cyclone = sargeo_df[sargeo_df["sar_inter"] == inter_path+".nc"]["cyclone"].iloc[0]
    irwin_path = os.path.join(sargeo_path,cyclone,"IRWIN")
    geo_path = os.path.join(irwin_path,sargeo_df[sargeo_df["sar_inter"] == inter_path+".nc"]["fichier"].iloc[0])

    return geo_path

def plot_sar_ir(date="20241006t092222",path_sar_folder="/scale/user/mtannaou/alternance/donnees_sar_aeqd",show=True):

    sar_aeqd_file = [f for f in os.listdir(path_sar_folder) if date in f][0]

    geo_path = get_geo_path(sar_aeqd_file)

    ds_sar = xr.open_dataset(geo_path, decode_timedelta=True)
    ds_geo = ds_sar
    ds_xy = xr.open_dataset(os.path.join(path_sar_folder,sar_aeqd_file))

    # === Chargement des colormaps ===
    cmap_ir = CMAP.cira_ir(units="celsius")
    cmap_sar = CMAP.cmap_sar()

    # === Données ===
    wind = ds_xy["owiWindSpeed"] * 1.94384   # m/s → kt
    ir = ds_geo["IRWIN"].sel(t_rel=0) - 273.15   # K → °C
    xs, ys = ds_xy["x_sar"], ds_xy["y_sar"]

    title = os.path.basename(geo_path).split("-")[4]
    date_fmt = f"{title[0:4]}-{title[4:6]}-{title[6:8]} {title[9:11]}:{title[11:13]}:{title[13:15]}"

    # === Figure ===
    fig = plt.figure(figsize=(8, 9))

    # Grille bien équilibrée
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
        hspace=0.07, wspace=0.07, bottom=0.08, top=0.93, left=0.08, right=0.95
    )

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])

    # === PANEL (0,0) : SAR wind ===
    p1 = ax11.pcolormesh(xs, ys, wind, cmap=cmap_sar, shading='auto', vmin=0, vmax=150)
    ax11.set_xlim(-300, 300)
    ax11.set_ylim(-300, 300)
    ax11.set_ylabel("y (km)")
    ax11.set_aspect('equal')
    ax11.set_title("SAR-derived wind speed", fontsize=12)
    ax11.set_xticks([])
    ax11.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax11.axvline(0, color='black', linestyle='--', linewidth=0.7)

    # === PANEL (0,1) : IR Brightness ===
    x_ir, y_ir = ds_geo["x"], ds_geo["y"]
    p2 = ax12.pcolormesh(x_ir, y_ir, ir, cmap=cmap_ir, shading='auto', vmin=-100, vmax=40)
    ax12.set_xlim(-300, 300)
    ax12.set_ylim(-300, 300)
    ax12.set_aspect('equal')
    ax12.set_title("Brightness Temperature", fontsize=12)
    ax12.set_xticks([])
    ax12.set_yticks([])
    ax12.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax12.axvline(0, color='black', linestyle='--', linewidth=0.7)

    # === PANEL (1,0) : SAR + IR contours ===
    p3 = ax21.pcolormesh(xs, ys, wind, shading='auto', cmap=cmap_sar, vmin=0, vmax=150)
    ax21.contour(x_ir, y_ir, ir, cmap=cmap_ir, linewidths=0.8, levels=10)
    ax21.set_xlim(-300, 300)
    ax21.set_ylim(-300, 300)
    ax21.set_xlabel("x (km)")
    ax21.set_ylabel("y (km)")
    ax21.set_aspect('equal')
    ax21.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax21.axvline(0, color='black', linestyle='--', linewidth=0.7)

    # === PANEL (1,1) : IR + SAR contours ===
    p4 = ax22.pcolormesh(x_ir, y_ir, ir, cmap=cmap_ir, shading='auto', vmin=-100, vmax=40)
    ax22.contour(xs, ys, wind, cmap=cmap_sar, linewidths=0.8, levels=10)
    ax22.set_xlim(-300, 300)
    ax22.set_ylim(-300, 300)
    ax22.set_xlabel("x (km)")
    ax22.set_aspect('equal')
    ax22.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax22.axvline(0, color='black', linestyle='--', linewidth=0.7)
    ax22.set_yticks([])

    # === AXES : TICKS MAJEURS / MINEURS ===
    major_ticks = np.arange(-300, 301, 100)
    minor_ticks = np.arange(-300, 301, 20)

    for ax in [ax11, ax12, ax21, ax22]:
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        ax.tick_params(axis='both', which='major', length=6, width=1.0, direction='inout')
        ax.tick_params(axis='both', which='minor', length=3, width=0.6, direction='inout')

        # Afficher seulement les labels majeurs (100 et 200)
        ax.set_xticklabels([str(t) if abs(t) in [100, 200] else "" for t in major_ticks])
        ax.set_yticklabels([str(t) if abs(t) in [100, 200] else "" for t in major_ticks])

    # === TITRE GLOBAL ===
    plt.suptitle(
        f"SAR vs IR comparison – {date_fmt} OLD Version\n",
        fontsize=14, fontweight="bold"
    )

    # === COLORBARS — fixées et centrées ===
    cbar_sar = fig.colorbar(
        p1, ax=[ax11, ax21],
        orientation='horizontal',
        fraction=0.046, pad=0.08, anchor=(0.5, -2.0)
    )
    cbar_sar.set_label("SAR-derived wind speed (kt)")

    cbar_ir = fig.colorbar(
        p2, ax=[ax12, ax22],
        orientation='horizontal',
        fraction=0.046, pad=0.08, anchor=(0.5, -2.0)
    )
    cbar_ir.set_label("Infrared brightness temperature (°C)")

    # === ALIGNEMENT PARFAIT ===
    fig.align_ylabels([ax11, ax21])
    fig.align_xlabels([ax21, ax22])
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Créer un dossier de sortie
    os.makedirs("/scale/user/mtannaou/alternance/src/visualisation/outputs", exist_ok=True)

    
    if show : 
        plt.show()
    else : 
        # Sauvegarder l’image
        output_file = f"/scale/user/mtannaou/alternance/src/visualisation/outputs/sar_ir_{date_fmt}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        os.sync()
        os.system(f"touch {os.path.dirname(output_file)}")
        print(f"✅ Figure sauvegardée : {output_file}")

if __name__ == "__main__":
    plot_sar_ir(date="20241007t20",show=False)

