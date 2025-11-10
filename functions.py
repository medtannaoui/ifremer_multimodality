import numpy as np
from scipy.ndimage import gaussian_filter
from datetime import datetime,timedelta
import os
import xarray as xr
import pyproj
from matplotlib import pyplot as plt


def convert_sar_to_xy(sar_file, sargeo_df, output_dir):
    """
    Convertit un fichier SAR en coordonnées (x, y) centrées sur le cyclone.
    Ajoute les champs x_sar, y_sar et sauvegarde un nouveau NetCDF.

    sar_file : str → chemin du fichier SAR .nc
    sargeo_df : DataFrame → contient les centres (lat_centre, lon_centre)
    output_dir : str → dossier de sortie
    for file_path in sargeo["sar_path"]: 
    try: 
        fct.convert_sar_to_xy(file_path, sargeo, output_dir) 
    except Exception as e: 
        print(f"❌ Erreur pour {file_path}: {e}")
    """

    # 1️⃣ Charger le fichier SAR
    ds = xr.open_dataset(sar_file)
    filename = os.path.basename(sar_file).split(".nc")[0]
    
    match = sargeo_df[sargeo_df["sar_inter"].str.contains(filename, case=False, na=False)]

    if match.empty:
        print(f"⚠️ Aucun centre trouvé pour {filename}")
        return

    lat_centre = float(match.iloc[0]["lat_centre"])
    lon_centre = float(match.iloc[0]["lon_centre"])
    if lon_centre > 180:
        lon_centre -= 360

    # 3️⃣ Définir la projection AEQD centrée sur le cyclone
    proj_geo = pyproj.Proj(proj='latlong', datum='WGS84')
    proj_xy = pyproj.Proj(proj='aeqd', lon_0=lon_centre, lat_0=lat_centre, datum='WGS84', units='km')

    # 4️⃣ Calculer les x, y à partir des lat/lon SAR
    x, y = pyproj.transform(
        proj_geo, proj_xy,
        ds["owiLon"].values,
        ds["owiLat"].values,
        always_xy=True
    )

    # 5️⃣ Ajouter les champs x_sar, y_sar
    ds["x_sar"] = (("owiAzSize", "owiRaSize"), x)
    ds["y_sar"] = (("owiAzSize", "owiRaSize"), y)

    # 6️⃣ Sauvegarder le nouveau fichier
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{filename}_aeqd.nc")
    #recuperer l indice de la ligne courante dans le dataframe
    current_index = sargeo_df[sargeo_df["sar_inter"].str.contains(filename, case=False, na=False)].index[0]
    #mettre a jour le chemin du fichier modifié dans le dataframe
    sargeo_df.at[current_index, "sar_xy"] = out_file
    sargeo_df.to_csv("excels/SARGEO_SAR.csv", index=False)  # Mettre à jour le CSV avec le nouveau chemin
    ds.to_netcdf(out_file)
    


    ds.close()


def center_and_mean_calm_pixels(x, y, WS, seuil_strong, vent_oeil, n_points=10):
    """
    1) Trouve le barycentre des pixels forts (WS >= seuil_strong)
    2) Cherche les n_points pixels calmes (WS < vent_oeil) les plus proches
    3) Retourne leur moyenne (xo, yo)

    Sorties :
      xc, yc  : centre des vents forts
      xo, yo  : moyenne des n pixels calmes les + proches
    """

    # --- 1) centre des vents forts ---
    mask_strong = (WS >= seuil_strong) & np.isfinite(WS)

    if not np.any(mask_strong):
        return None, None, None, None

    if x.ndim == 2:
        xs = x[mask_strong]
        ys = y[mask_strong]
    else:
        iy, ix = np.where(mask_strong)
        xs = x[ix]
        ys = y[iy]

    xc = float(np.mean(xs))
    yc = float(np.mean(ys))

    # --- 2) pixels calmes ---
    mask_calm = (WS < vent_oeil) & np.isfinite(WS)
    if not np.any(mask_calm):
        return xc, yc, None, None

    if x.ndim == 2:
        xcand = x[mask_calm]
        ycand = y[mask_calm]
    else:
        iy2, ix2 = np.where(mask_calm)
        xcand = x[ix2]
        ycand = y[iy2]

    # --- 3) calcul distance par rapport à (xc,yc) ---
    d2 = (xcand - xc)**2 + (ycand - yc)**2
    idx = np.argsort(d2)   # tri des distances

    # on prend les n_points premiers
    k = min(n_points, len(idx))
    sel = idx[:k]

    xo = float(np.mean(xcand[sel]))
    yo = float(np.mean(ycand[sel]))

    return xc, yc, xo, yo



def is_within_deltamin(date1_str, date2_str, delta=10):
    """
    Vérifie si date1 est dans l'intervalle [date2 - 10 min, date2 + 10 min].
    
    Paramètres :
        date1_str (str) : première date, ex '2019-09-20T22:19:52.000'
        date2_str (str) : seconde date, ex '2019-09-20T22:25:00.000'
        delta : 
    
    Retourne :
        bool : True si date1 est dans l'intervalle, sinon False
    """
    fmt = "%Y-%m-%dT%H:%M:%S.%f"
    
    # Conversion en datetime
    d1 = datetime.strptime(date1_str, fmt)
    d2 = datetime.strptime(date2_str, fmt)
    
    # Création de l'intervalle ±delta minutes
    delta = timedelta(minutes=delta)
    
    return (d2 - delta) <= d1 <= (d2 + delta)


