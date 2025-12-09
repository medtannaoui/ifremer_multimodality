# # This script is used to reproject the SAR data onto the SARGEO grid
from importlib import reload
import os
import numpy as np
import pandas as pd
import xarray as xr
from SAR_NESDIS import NESDIS 
import shutil

tcprimed_path = "/scale/project/ifremer-isi-jumeaunumerique/TC_PRIMED_DATASET/v01r01/final"   #tcprimed path
tcprimed_path_preliminary = "/scale/project/ifremer-isi-jumeaunumerique/TC_PRIMED_DATASET/v01r01/preliminary"

def convert_sar_to_xy(sar_file, sargeo_df, output_dir, max_r=300, dxy=2, center="storm"):
    """
    Convertit un fichier SAR en coordonnées (x, y) centrées sur le cyclone
    en utilisant la méthode NESDIS.aeqd().
    """
    
    
    # Charger le fichier SAR original
    ds = xr.open_dataset(sar_file)
    # print(ds.data_vars)
    filename = os.path.basename(sar_file).split(".nc")[0][:-6]

    # Trouver le centre correspondant dans le DataFrame
    #vmax,analysis_vmax,analysis_center_quality_flag,analysis_rmax
    match = sargeo_df[sargeo_df["sar_inter"].str.contains(filename, case=False, na=False)]
    if match.empty:
        print(f"⚠️ Aucun centre trouvé pour {filename}")
        return
    cyclone_name = match.iloc[0]["cyclone"]
    lat_centre = float(match.iloc[0]["eye_center_lat"]) if center == "eye" else    float(match.iloc[0]["lat_centre"])       #use the eye_center

    lon_centre = float(match.iloc[0]["eye_center_lon"]) if center == "eye" else float(match.iloc[0]["lon_centre"])
    lon_centre = (lon_centre + 180) % 360 - 180  # longitude center normalisation

    print(f"📡 Traitement du fichier {filename} — centre cyclone : ({lat_centre}, {lon_centre})")

    # Vérifier la présence des variables latitude / longitude
    if "owiLon" not in ds or "owiLat" not in ds:
        raise ValueError(f"❌ Variables 'owiLon' et 'owiLat' absentes dans {filename}")


    ds["owiLon"] = (ds["owiLon"] + 180) % 360 - 180    #longitude normalisation

    # application of the aeqd projection
    sar_aeqd, _ = NESDIS.aeqd(
        sar_org=ds,
        clon=lon_centre,
        clat=lat_centre,
        max_r=max_r,         # maximal radius
        dxy=dxy,           # résolution (in km)
        kind="nearest",    # nearest or linear
        varnames=["owiWindSpeed","owiWindSpeed_co","owiWindDirection_co","owiIncidenceAngle"],
        include_org=False,
        match = match
    )

    # Sauvegarder le nouveau fichier NetCDF
    os.makedirs(output_dir, exist_ok=True)
    # env = False
    if not os.path.isdir(os.path.join(output_dir,cyclone_name)):
        
        year_cyclone = cyclone_name[4:]
        num_cyclone = cyclone_name[2:4]
        bassin_cyclone = cyclone_name[0:2]
        cyclone_path_primed = os.path.join(tcprimed_path, str(year_cyclone), bassin_cyclone.upper(), str(num_cyclone) )
        path_env = None
        cyclone_path_primed = cyclone_path_primed if os.path.isdir(cyclone_path_primed) else os.path.join(tcprimed_path_preliminary, str(year_cyclone), bassin_cyclone.upper(), str(num_cyclone) )
        if os.path.isdir(cyclone_path_primed):              # if it exists in tcprimed dataset
            os.makedirs(os.path.join(output_dir,cyclone_name),exist_ok=True)
            for path_i in  os.listdir(cyclone_path_primed):
                if "env" in path_i :
                    # env = True
                    path_env = os.path.join(cyclone_path_primed, path_i)
                    break
            
            shutil.copy(path_env, os.path.join(output_dir,cyclone_name,path_i))
        else : #if does not exist in tcprimed
            cyclone_name = cyclone_name+"no_env"
            os.makedirs(os.path.join(output_dir,cyclone_name),exist_ok=True)



    out_file = os.path.join(os.path.join(output_dir,cyclone_name), f"{filename}_aeqd.nc")
    sar_aeqd.to_netcdf(out_file)
    ds.close()

    # Mettre à jour ton CSVa ton avis 
    current_index = match.index[0]
    sargeo_df.at[current_index, "sar_xy"] = out_file
    sargeo_df.to_csv("excels/SARGEO_SAR_v5.csv", index=False)

    print(f"✅ Fichier reprojeté sauvegardé : {out_file}")


def main(dxy=2,max_r=300,center="storm"):
    output_dir = "/scale/user/mtannaou/alternance/donnees_sar_aeqd_v4"
    sargeo = pd.read_csv("excels/SARGEO_SAR_v5.csv")

    # test only for al122024
    # sargeo = sargeo[sargeo["cyclone"] == "al122024"]
    sargeo = sargeo[sargeo["lat_centre"] != 0]

    for file_path in sargeo["sar_path"]:
        try:
            convert_sar_to_xy(file_path, sargeo, output_dir, max_r=max_r, dxy=dxy, center=center)
        except Exception as e:
            print(f"❌ Erreur pour {file_path}: {e}")

    print(f"📁 Nombre de fichiers reprojetés : {len(os.listdir(output_dir))}")


if __name__ == "__main__":
    main(2,300, center="eye")










































# import os
# import xarray as xr
# import pyproj
# from matplotlib import pyplot as plt
# import pandas as pd

# run = True

# def convert_sar_to_xy(sar_file, sargeo_df, output_dir):
#     """
#     Converts a SAR file into (x, y) coordinates centered on the cyclone.
#     Adds the x_sar and y_sar fields and saves a new NetCDF file.

#     sar_file : str → path to the SAR .nc file
#     sargeo_df : DataFrame → contains the cyclone centers (lat_center, lon_center)
#     output_dir : str → output directory

#     for file_path in sargeo["sar_path"]:
#         try:
#             fct.convert_sar_to_xy(file_path, sargeo, output_dir)
#         except Exception as e:
#             print(f"❌ Error for {file_path}: {e}")
#     """

#     # Charger le fichier SAR
#     ds = xr.open_dataset(sar_file)
#     filename = os.path.basename(sar_file).split(".nc")[0]
    
#     match = sargeo_df[sargeo_df["sar_inter"].str.contains(filename, case=False, na=False)]

#     if match.empty:
#         print(f"⚠️ Aucun centre trouvé pour {filename}")
#         return

#     lat_centre = float(match.iloc[0]["lat_centre"])
#     lon_centre = float(match.iloc[0]["lon_centre"])
#     # (ds[coords["lon"]] + 180) % 360 - 180
#     lon_centre = (lon_centre + 180) % 360 -180    # longitude verification
#     # if lon_centre > 180:
#     #     lon_centre -= 360

#     # Use the AEQD projection 
#     proj_geo = pyproj.Proj(proj='latlong', datum='WGS84')
#     proj_xy = pyproj.Proj(proj='aeqd', lon_0=lon_centre, lat_0=lat_centre, datum='WGS84', units='km')

#     # caluculate x and y with pyproj
#     ds["owiLon"].values = (ds["owiLon"].values + 180) % 360 - 180
#     x, y = pyproj.transform(
#         proj_geo, proj_xy,
#         ds["owiLon"].values,
#         ds["owiLat"].values,
#         always_xy=True
#     )

#     # add x_sar and y_sar
#     ds["x_sar"] = (("owiAzSize", "owiRaSize"), x)
#     ds["y_sar"] = (("owiAzSize", "owiRaSize"), y)

#     # save the new file
#     os.makedirs(output_dir, exist_ok=True)
#     out_file = os.path.join(output_dir, f"{filename}_aeqd.nc")
#     #recuperer l indice de la ligne courante dans le dataframe
#     current_index = sargeo_df[sargeo_df["sar_inter"].str.contains(filename, case=False, na=False)].index[0]
#     # update the new path of the file in the dataframe
#     sargeo_df.at[current_index, "sar_xy"] = out_file
#     sargeo_df.to_csv("excels/SARGEO_SAR_v1.csv", index=False)  # updatethe csv file
#     ds.to_netcdf(out_file)

#     ds.close()


# def main():
#     output_dir = "/scale/user/mtannaou/alternance/donnees_sar_changes_update"
#     sargeo = pd.read_csv("excels/SARGEO_SAR.csv")

#     for file_path in sargeo["sar_path"]: 
#         try: 
#             convert_sar_to_xy(file_path, sargeo, output_dir) 
#         except Exception as e: 
#             print(f"❌ Erreur pour {file_path}: {e}")

#     print(len(os.listdir(output_dir)))


# if run : 
#     main()