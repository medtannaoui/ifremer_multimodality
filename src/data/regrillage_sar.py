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

def convert_sar_to_xy(sar_file, i, row, sargeo_df, output_dir, max_r=300, dxy=2, center="storm"):
    """
    Convertit un fichier SAR en coordonnées (x, y) centrées sur le cyclone
    en utilisant la méthode NESDIS.aeqd().
    """
    
    
    # Charger le fichier SAR original
    ds = xr.open_dataset(sar_file)
    # print(ds.data_vars)
    filename = os.path.basename(sar_file).split(".nc")[0][:-6]

    # Trouver le centre correspondant dans le DataFrame
    # vmax,analysis_vmax,analysis_center_quality_flag,analysis_rmax
    match = row
    if pd.isna(match.get("lat_centre")):         #si lat_centre est null
        print(f"⚠️ Aucun centre trouvé pour {filename}")
        return
    cyclone_name = match.get("cyclone")
    lat_centre = float(match.get("eye_center_lat")) if center == "eye" else    float(match.get("lat_centre"))       #use the eye_center

    lon_centre = float(match.get("eye_center_lon")) if center == "eye" else float(match.get("lon_centre"))
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
    current_index = i
    sargeo_df.at[current_index, "sar_xy"] = out_file
    sargeo_df.to_csv("excels/SARGEO_SAR_v5.csv", index=False)

    print(f"✅ Fichier reprojeté sauvegardé : {out_file}")


def main(dxy=2,max_r=300,center="storm"):
    output_dir = "/scale/user/mtannaou/alternance/donnees_sar_aeqd_3km"
    sargeo = pd.read_csv("excels/SARGEO_SAR_v00r00_09_janvier_v1.csv")

    # test only for al122024
    # sargeo = sargeo[sargeo["cyclone"] == "al122024"]
    sargeo = sargeo[sargeo["lat_centre"] != 0]

    for i, row in sargeo.iterrows():
        file_path =  row["sar_path"]
        try:
            convert_sar_to_xy(file_path, i ,row, sargeo, output_dir, max_r=max_r, dxy=dxy, center=center)
        except Exception as e:
            print(f"❌ Erreur pour {file_path}: {e}")

    print(f"📁 Nombre de fichiers reprojetés : {len(os.listdir(output_dir))}")


if __name__ == "__main__":
    main(3,500, center="storm")