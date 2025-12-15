"""
This script creates colocated data packages (IR - MW - SAR) in .pkl format.

The output file contains 4 datasets:
    - IR from TC_PRIMED
    - MW from TC_PRIMED
    - IR from SARGEO
    - SAR AEQD product
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



import os
import ast
import xarray as xr
import pandas as pd
import netCDF4 as nc
import pickle as pkl
from datetime import datetime


# ============================================================
# ========================== CONFIG ==========================
# ============================================================

CSV_COLOCATION = "/scale/user/mtannaou/alternance/excels/colocates_sargeo_primed_90min_v1.csv"
CSV_SARGEO_SAR = "/scale/user/mtannaou/alternance/excels/SARGEO_SAR_v4.csv"

PATH_TCPRIMED = "/scale/project/ifremer-isi-jumeaunumerique/TC_PRIMED_DATASET/v01r01/final"
PATH_SARGEO   = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v00r00/cyclobs"

OUTPUT_DIR    = "/scale/user/mtannaou/alternance/src/IR_MW_to_SAR/data"


# ============================================================
# ============ MAIN PROCESSING FUNCTION ======================
# ============================================================

import os
import xarray as xr
import pandas as pd
import netCDF4 as nc
import pickle as pkl
import numpy as np
from datetime import datetime
import gc


def create_coloc_primed_sargeo(output_dir=OUTPUT_DIR):


    os.makedirs(output_dir, exist_ok=True)

    df_sargeo_sar = pd.read_csv(CSV_SARGEO_SAR)
    df_coloc      = pd.read_csv(CSV_COLOCATION)

    df_coloc = df_coloc[df_coloc["count"] != 0].reset_index(drop=True)

    netcdf_paths = []
    n_saved = 0

    for idx, row in df_coloc.iterrows():

        cyclone_id = row["cyclone_id"]
        basin      = cyclone_id[:2].upper()
        num        = cyclone_id[2:4]
        year       = cyclone_id[4:]

        tcprimed_path = os.path.join(PATH_TCPRIMED, str(year), basin, str(num))

        tc_files = row["path_tcprimed"].split("]")[0].split("[")[-1].split(",")    #verified
        diffs_time = row["diff_time"]
       
        # print(diffs_time)
     
        
        
    
        for k, nc_file in enumerate(tc_files):
            
            
            try:
                tc_time = datetime.strptime(
                    row["date_tcprimed"].replace(" ","")
                    .split("]")[0].split("[")[-1].split(",")[k],
                    "'%Y-%m-%dT%H:%M:%S.000'"
                )
               

                sargeo_time = datetime.strptime(
                    row["date_sargeo"].replace(" ", ""),
                    "%Y-%m-%dT%H:%M:%S.000"
                )
                

                file_name = nc_file[1:-1].split("'")[-1]
                
                full_nc = os.path.join(tcprimed_path, file_name)
                # print("Loading:", full_nc)

                with nc.Dataset(full_nc) as nc_data:
                    groups = list(nc_data.groups.keys())
                    
                    if "passive_microwave" not in groups:
                        print("nop passive microwave group")
                        continue

                    ds_mw = xr.open_dataset(full_nc, group=f"passive_microwave/S1")
                    ds_ir = xr.open_dataset(full_nc, group= "infrared")
                    

                    mw_keys = list(ds_mw.data_vars.keys())[5:]
                    lat_mw = ds_mw["latitude"]
                    lon_mw = ds_mw["longitude"]
                    
                   
                    mw_data = {k: ds_mw[k].values for k in mw_keys}
                   
                    

                    if "IRWIN" not in ds_ir.data_vars:
                        print("no IRWIN group")
                        continue
                    
                    ir_tc = ds_ir["IRWIN"].values
                    
                   
                
                    
                    lon_ir  = ds_ir["longitude"].values
                    lat_ir  = ds_ir["latitude"].values

                
                sargeo_base = row["path_sargeo"].split("_ll_gd")[0][:-7]

                sar_match = df_sargeo_sar[
                    df_sargeo_sar["fichier"].str.contains(sargeo_base)
                ]
                
                if len(sar_match) == 0:
                   
                    
                    continue
               
                
                sar_file = sar_match["sar_xy"].values[0]
                lat_centre = sar_match["lat_centre"]
                lon_centre = sar_match["lon_centre"]
                eye_center_lat = sar_match["eye_center_lat"]
                eye_center_lon = sar_match["eye_center_lon"]
              

                with xr.open_dataset(sar_file) as ds_sar:
                    
                    
                    sar_data = ds_sar["owiWindSpeed"].values
                    x_sar = ds_sar["x_sar"].values
                    y_sar = ds_sar["y_sar"].values


                
                sargeo_filename = sar_match["fichier"].values[0]
           
                sargeo_ir_path = os.path.join(
                    PATH_SARGEO, cyclone_id, "IRWIN", sargeo_filename
                )
               

                with xr.open_dataset(sargeo_ir_path) as ds_sargeo:
                    
                    ir_sargeo = ds_sargeo["IRWIN"].values
                    

            
                diff_sec = int(abs(tc_time - sargeo_time).total_seconds())
                # print(diff_sec)
                # print(lon_centre.values)
                ds_out = xr.Dataset(
                    data_vars = {
                        **{f"mw_{k}": (("scan", "pixel"), v) for k, v in mw_data.items()},
                        "ir_tcprimed": (("ir_tc_height", "ir_tc_width"), ir_tc),
                        "sar_aeqd": (("sar_aeqd_height", "sar_aeqd_width"), sar_data),
                        "ir_sargeo": (("channel","ir_sargeo_height", "ir_sargeo_width"), ir_sargeo),
                        "x_sar": (("x_sar",), x_sar),
                        "y_sar": (("y_sar",), y_sar),
                        "lat_mw": (("scan","pixel"),lat_mw.values),
                        "lon_mw":(("scan","pixel"),lon_mw.values),
                        "x_mw":(("scan","pixel"),ds_mw["x"].values),
                        "y_mw":(("scan","pixel"),ds_mw["y"].values),
                        "x_ir":(("scan_ir","pixel_ir"),ds_ir["x"].values),
                        "y_ir":(("scan_ir","pixel_ir"),ds_ir["y"].values),
                        "lon_ir_primed":(("scan_ir","pixel_ir"),lon_ir),
                        "lat_ir_primed":(("scan_ir","pixel_ir"),lat_ir),
                        "lon_centre": ((), lon_centre.values[0]),
                        "lat_centre": ((), lat_centre.values[0]),
                        "eye_center_lat": ((), eye_center_lat.values[0]),
                        "eye_center_lon": ((), eye_center_lon.values[0])
                        
                    },

                    attrs={
                        "cyclone_id": cyclone_id,
                        "tcprimed_time": str(tc_time),
                        "sargeo_time": str(sargeo_time),
                        "diff_secondes": str(diff_sec)
                    },
                )

                # diff_min = int(abs(tc_time - sargeo_time).total_seconds())
                # out_name = f"{cyclone_id}_{tc_time}_diff_{diff_min}min.nc"
                # out_path = os.path.join(output_dir, out_name)

                # ds_out.to_netcdf(out_path)
                # ds_out.close()

                netcdf_paths.append(ds_out)
                n_saved += 1

        
            except Exception as e:
                print("❌ Erreur sur un triplet :", e)
                continue
        
        # if idx == 50 : 
        #     break
        
        

    pkl_path = os.path.join(output_dir, "coloc_primed_sargeo_v1.pkl")
    with open(pkl_path, "wb") as f:
        pkl.dump(netcdf_paths, f)

    print(f"\n✅ {n_saved} triplets sauvegardés")
    print(f"📦 Pickle enregistré : {pkl_path}")


# ============================================================
# =========================== RUN ============================
# ============================================================

if __name__ == "__main__":
    create_coloc_primed_sargeo()
    print("\n✨ All colocated PKL files generated successfully.")
