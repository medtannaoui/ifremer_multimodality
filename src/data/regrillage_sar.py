# This script is used to reproject the SAR data onto the SARGEO grid
import os
import xarray as xr
import pyproj
from matplotlib import pyplot as plt
import pandas as pd

run = True

def convert_sar_to_xy(sar_file, sargeo_df, output_dir):
    """
    Converts a SAR file into (x, y) coordinates centered on the cyclone.
    Adds the x_sar and y_sar fields and saves a new NetCDF file.

    sar_file : str → path to the SAR .nc file
    sargeo_df : DataFrame → contains the cyclone centers (lat_center, lon_center)
    output_dir : str → output directory

    for file_path in sargeo["sar_path"]:
        try:
            fct.convert_sar_to_xy(file_path, sargeo, output_dir)
        except Exception as e:
            print(f"❌ Error for {file_path}: {e}")
    """

    # Charger le fichier SAR
    ds = xr.open_dataset(sar_file)
    filename = os.path.basename(sar_file).split(".nc")[0]
    
    match = sargeo_df[sargeo_df["sar_inter"].str.contains(filename, case=False, na=False)]

    if match.empty:
        print(f"⚠️ Aucun centre trouvé pour {filename}")
        return

    lat_centre = float(match.iloc[0]["lat_centre"])
    lon_centre = float(match.iloc[0]["lon_centre"])
    # (ds[coords["lon"]] + 180) % 360 - 180
    lon_centre = (lon_centre + 180) % 360 -180    # longitude verification
    # if lon_centre > 180:
    #     lon_centre -= 360

    # Use the AEQD projection 
    proj_geo = pyproj.Proj(proj='latlong', datum='WGS84')
    proj_xy = pyproj.Proj(proj='aeqd', lon_0=lon_centre, lat_0=lat_centre, datum='WGS84', units='km')

    # caluculate x and y with pyproj
    ds["owiLon"].values = (ds["owiLon"].values + 180) % 360 - 180
    x, y = pyproj.transform(
        proj_geo, proj_xy,
        ds["owiLon"].values,
        ds["owiLat"].values,
        always_xy=True
    )

    # add x_sar and y_sar
    ds["x_sar"] = (("owiAzSize", "owiRaSize"), x)
    ds["y_sar"] = (("owiAzSize", "owiRaSize"), y)

    # save the new file
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{filename}_aeqd.nc")
    #recuperer l indice de la ligne courante dans le dataframe
    current_index = sargeo_df[sargeo_df["sar_inter"].str.contains(filename, case=False, na=False)].index[0]
    # update the new path of the file in the dataframe
    sargeo_df.at[current_index, "sar_xy"] = out_file
    sargeo_df.to_csv("excels/SARGEO_SAR_v1.csv", index=False)  # updatethe csv file
    ds.to_netcdf(out_file)

    ds.close()


def main():
    output_dir = "/scale/user/mtannaou/alternance/donnees_sar_changes_update"
    sargeo = pd.read_csv("excels/SARGEO_SAR.csv")

    for file_path in sargeo["sar_path"]: 
        try: 
            convert_sar_to_xy(file_path, sargeo, output_dir) 
        except Exception as e: 
            print(f"❌ Erreur pour {file_path}: {e}")

    print(len(os.listdir(output_dir)))


if run : 
    main()