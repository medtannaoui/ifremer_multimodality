"""
This script creates colocated data packages (IR - MW - SAR) in .pkl format.

The output file contains 4 datasets:
    - IR from TC_PRIMED
    - MW from TC_PRIMED
    - IR from SARGEO
    - SAR AEQD product
"""

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

CSV_COLOCATION = "/scale/user/mtannaou/alternance/excels/colocates_sargeo_primed_v1.csv"
CSV_SARGEO_SAR = "/scale/user/mtannaou/alternance/excels/SARGEO_SAR_v1.csv"

PATH_TCPRIMED = "/scale/project/ifremer-isi-jumeaunumerique/TC_PRIMED_DATASET/v01r01/final"
PATH_SARGEO   = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v01r02/cyclobs"
PATH_SAR_AEQD = "/scale/user/mtannaou/alternance/donnees_sar_aeqd"

OUTPUT_DIR    = "/scale/user/mtannaou/alternance/src/data_coloc_pkl"


# ============================================================
# ============ MAIN PROCESSING FUNCTION ======================
# ============================================================

def create_colocation_pkl(output_path: str = OUTPUT_DIR):
    """
    Creates pkl files containing colocated IR, MW, SARGEO IR, and AEQD SAR datasets.

    Output .pkl structure:
        [
            [mw_x, mw_y, mw_value],                   # Microwave
            [ir_tcprimed, ir_x, ir_y],                # IR from TC_PRIMED
            [sar_x, sar_y, sar_windspeed],            # SAR AEQD
            [sargeo_ir, x, y, storm_lat, storm_lon]   # SARGEO IR
        ]
    """

    # Load metadata CSV files
    df_sargeo_sar = pd.read_csv(CSV_SARGEO_SAR)
    df_coloc      = pd.read_csv(CSV_COLOCATION)

    # Only keep rows with at least one matched product
    df_coloc = df_coloc[df_coloc["count"] != 0].reset_index(drop=True)

    for idx, row in df_coloc.iterrows():

        cyclone_id    = row["cyclone_id"]
        basin         = cyclone_id[:2].upper()
        cyclone_num   = cyclone_id[2:4]
        cyclone_year  = cyclone_id[4:]

        # Path to TC_PRIMED cyclone folder
        tcprimed_cyclone_path = os.path.join(PATH_TCPRIMED, str(cyclone_year), basin, str(cyclone_num))

        # Extract all TC_PRIMED NetCDF file names
        tcprimed_files = row["path_tcprimed"].split("]")[0].split("[")[-1].split(",")
        print(f"→ Number of TC_PRIMED files found: {len(tcprimed_files)}")

        for dt_index, nc_file in enumerate(tcprimed_files):

            # Parse aligned timestamps
            tcprimed_timestamp = datetime.strptime(
                row["date_tcprimed"].replace(" ", "").split("]")[0].split("[")[-1].split(",")[dt_index],
                "'%Y-%m-%dT%H:%M:%S.000'"
            )
            sargeo_timestamp = datetime.strptime(row["date_sargeo"].replace(" ", ""), "%Y-%m-%dT%H:%M:%S.000")

            print(f"TC_PRIMED datetime → {tcprimed_timestamp}")

            # Path to the TC_PRIMED NetCDF file
            file_name = nc_file[1:-1].split("'")[-1]
            full_nc_path = os.path.join(tcprimed_cyclone_path, file_name)
            print("Loading:", full_nc_path)

            # Load NetCDF and extract the correct groups
            nc_data = nc.Dataset(full_nc_path)
            groups = list(nc_data.groups.keys())

            ds_mw = xr.open_dataset(full_nc_path, group=f"{groups[2]}/S1")  # Microwave group
            ds_ir_tc = xr.open_dataset(full_nc_path, group=groups[-1])      # Infrared group

            coloc_data = []

            # ======================================================
            # 1) MICROWAVE DATA (x, y, value)
            # ======================================================
            mw_key = list(ds_mw.data_vars.keys())[5]
            coloc_data.append([
                ds_mw["x"].values,
                ds_mw["y"].values,
                ds_mw[mw_key].values
            ])

            # ======================================================
            # 2) IR FROM TC_PRIMED
            # ======================================================
            if "IRWIN" not in ds_ir_tc.data_vars:
                continue

            coloc_data.append([
                ds_ir_tc["IRWIN"].values,
                ds_ir_tc["x"].values,
                ds_ir_tc["y"].values
            ])

            # ======================================================
            # 3) LOOKUP SAR AEQD PATH FROM SARGEO CSV
            # ======================================================
            sargeo_base = row["path_sargeo"].split("ll")[0][:-7]

            sar_path = df_sargeo_sar[df_sargeo_sar["fichier"].str.contains(sargeo_base)]["sar_xy"]
            if len(sar_path) == 0:
                continue

            sar_file = sar_path.values[0]

            # Skip if missing SARGEO center
            if df_sargeo_sar[df_sargeo_sar["sar_xy"] == sar_file]["lon_centre"].values == 0:
                continue

            # ======================================================
            # 4) SAR AEQD DATA (windspeed)
            # ======================================================
            ds_sar_aeqd = xr.open_dataset(sar_file)
            coloc_data.append([
                ds_sar_aeqd["x_sar"].values,
                ds_sar_aeqd["y_sar"].values,
                ds_sar_aeqd["owiWindSpeed"].values
            ])

            # ======================================================
            # 5) IR FROM SARGEO (t_rel = 0)
            # ======================================================
            sargeo_filename = df_sargeo_sar[df_sargeo_sar["sar_xy"] == sar_file]["fichier"].values[0]
            sargeo_irwin_path = os.path.join(PATH_SARGEO, cyclone_id, "IRWIN", sargeo_filename)

            ds_sargeo = xr.open_dataset(sargeo_irwin_path)
            ir_sargeo = ds_sargeo["IRWIN"].sel(t_rel=0).values

            coloc_data.append([
                ir_sargeo,
                ds_sargeo["x"].values,
                ds_sargeo["y"].values,
                ds_sargeo["storm_latitude"].sel(t_rel=0).values,
                ds_sargeo["storm_longitude"].sel(t_rel=0).values
            ])

            # ======================================================
            # Save result as pickle
            # ======================================================
            os.makedirs(output_path, exist_ok=True)

            diff_min = int(abs(tcprimed_timestamp - sargeo_timestamp).total_seconds() / 60)
            output_filename = f"{cyclone_id}--{row['date_sargeo']}---diff_minutes_{diff_min}.pkl"

            with open(os.path.join(output_path, output_filename), "wb") as f:
                pkl.dump(coloc_data, f)

            print(f"✓ Saved: {output_filename}")


# ============================================================
# =========================== RUN ============================
# ============================================================

if __name__ == "__main__":
    create_colocation_pkl()
    print("\n✨ All colocated PKL files generated successfully.")
