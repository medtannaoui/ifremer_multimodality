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

OUTPUT_DIR    = "/scale/user/mtannaou/alternance/src/sargeo_primed_colocs"


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

        tc_files = row["path_tcprimed"].split("]")[0].split("[")[-1].split(",")
        

        for k, nc_file in enumerate(tc_files):

            try:
                tc_time = datetime.strptime(
                    row["date_tcprimed"].replace(" ", "")
                    .split("]")[0].split("[")[-1].split(",")[k],
                    "'%Y-%m-%dT%H:%M:%S.000'"
                )

                sargeo_time = datetime.strptime(
                    row["date_sargeo"].replace(" ", ""),
                    "%Y-%m-%dT%H:%M:%S.000"
                )

                file_name = nc_file[1:-1].split("'")[-1]
                full_nc = os.path.join(tcprimed_path, file_name)
                print("Loading:", full_nc)

                with nc.Dataset(full_nc) as nc_data:
                    groups = list(nc_data.groups.keys())

                    ds_mw = xr.open_dataset(full_nc, group=f"{groups[2]}/S1")
                    ds_ir = xr.open_dataset(full_nc, group=groups[-1])

                    mw_keys = list(ds_mw.data_vars.keys())[5:]
                    mw_data = {k: ds_mw[k].values for k in mw_keys}

                    if "IRWIN" not in ds_ir:
                        continue

                    ir_tc = ds_ir["IRWIN"].values
                    x_ir  = ds_ir["x"].values
                    y_ir  = ds_ir["y"].values

                sargeo_base = row["path_sargeo"].split("ll")[0][:-7]
                sar_match = df_sargeo_sar[
                    df_sargeo_sar["fichier"].str.contains(sargeo_base)
                ]

                if len(sar_match) == 0:
                    continue

                sar_file = sar_match["sar_xy"].values[0]

                with xr.open_dataset(sar_file) as ds_sar:
                    sar_data = ds_sar["owiWindSpeed"].values
                    x_sar = ds_sar["x_sar"].values
                    y_sar = ds_sar["y_sar"].values

                sargeo_filename = sar_match["fichier"].values[0]
                sargeo_ir_path = os.path.join(
                    PATH_SARGEO, cyclone_id, "IRWIN", sargeo_filename
                )

                with xr.open_dataset(sargeo_ir_path, decode_timedelta=False) as ds_sargeo:
                    ir_sargeo = ds_sargeo["IRWIN"].values
                    x_sg = ds_sargeo["x"].values
                    y_sg = ds_sargeo["y"].values

            

                ds_out = xr.Dataset(
                    data_vars={
                        **{f"mw_{k}": (v) for k, v in mw_data.items()},
                        "ir_tcprimed": (ir_tc),
                        "sar_aeqd": (sar_data),
                        "ir_sargeo": (ir_sargeo),
                        "x_sar": ( x_sar),
                        "y_sar": (y_sar),
                        
                    },
                    attrs={
                        "cyclone_id": cyclone_id,
                        "tcprimed_time": str(tc_time),
                        "sargeo_time": str(sargeo_time),
                    },
                )

                diff_min = int(abs(tc_time - sargeo_time).total_seconds() / 60)
                out_name = f"{cyclone_id}_{tc_time}_diff_{diff_min}min.nc"
                out_path = os.path.join(output_dir, out_name)

                ds_out.to_netcdf(out_path)
                ds_out.close()

                netcdf_paths.append(out_path)
                n_saved += 1

                print("✅ Saved:", out_name)

            except Exception as e:
                print("❌ Erreur sur un triplet :", e)
                continue

    pkl_path = os.path.join(output_dir, "coloc_primed_sargeo.pkl")
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
