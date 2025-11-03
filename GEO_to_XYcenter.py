import os
import xarray as xr
import pyproj
import numpy as np

def convert_sar_to_xy(sar_file, sargeo_df, output_dir):
    """
    Convertit un fichier SAR en coordonnées (x, y) centrées sur le cyclone.
    Ajoute les champs x_sar, y_sar et sauvegarde un nouveau NetCDF.

    sar_file : str → chemin du fichier SAR .nc
    sargeo_df : DataFrame → contient les centres (lat_centre, lon_centre)
    output_dir : str → dossier de sortie
    """

    # 1️⃣ Charger le fichier SAR
    ds = xr.open_dataset(sar_file)
    filename = os.path.basename(sar_file).replace(".nc", "")
    #print(f"🔹 Traitement du fichier : {filename}")

    # 2️⃣ Trouver le centre du cyclone correspondant
    inter_key = filename.split("_centered")[0].split("_aeqd")[0]
    match = sargeo_df[sargeo_df["inter_list_sar"].str.contains(inter_key, case=False, na=False)]

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
    current_index = sargeo_df[sargeo_df["inter_list_sar"].str.contains(inter_key, case=False, na=False)].index[0]
    #mettre a jour le chemin du fichier modifié dans le dataframe
    sargeo_df.at[current_index, "sar_xy"] = out_file
    sargeo_df.to_csv("excels/SARGEO_modifiee.csv", index=False)  # Mettre à jour le CSV avec le nouveau chemin
    ds.to_netcdf(out_file)
    


    ds.close()
