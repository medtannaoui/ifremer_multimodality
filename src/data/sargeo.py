"""
This script scans the full SARGEO dataset and extracts metadata into a CSV file.

Output CSV columns:
    cyclone, channel, file, lat_center, lon_center, acquisition_time

It only keeps IRWIN and WV subfolders.
"""

import os
import xarray as xr
import pandas as pd


# ============================================================
# ========================== PATHS ============================
# ============================================================

SARGEO_ROOT = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v00r00/cyclobs"




#### Create sargeo_sar csv file ####
def create_sargeo_sar_csv():
    # --- Dossier racine SARGEO ---
    sargeo_path = SARGEO_ROOT
    # --- Liste des cyclones ---
    cyclones = os.listdir(sargeo_path)

    infos = []

    for cyclone in cyclones:
        cyclone_path = os.path.join(sargeo_path, cyclone)
        if not os.path.isdir(cyclone_path):
            continue

        # print(f"🌀 Traitement du cyclone : {cyclone}")

        # --- On regarde s'il y a les sous-dossiers IRWIN et WV ---
        for subdir in ["IRWIN", "WV"]:
            sub_path = os.path.join(cyclone_path, subdir)
            if not os.path.exists(sub_path):
                # print(f"  ⚠️ Pas de dossier {subdir} pour {cyclone}")
                continue

            # --- Liste des fichiers NetCDF ---
            nc_files = [f for f in os.listdir(sub_path) if f.endswith(".nc")]

            for nc_file in nc_files:
                file_path = os.path.join(sub_path, nc_file)
                try:
                    ds = xr.open_dataset(file_path)

                    # Vérifie que les champs existent
                    lat_centre = float(ds["storm_latitude"].values[4]) if "storm_latitude" in ds else None
                    lon_centre = float(ds["storm_longitude"].values[4]) if "storm_longitude" in ds else None
                    time = str(ds["sar_acquisition_time"].values) if "sar_acquisition_time" in ds else None

                    infos.append({
                        "cyclone": cyclone,
                        "canal": subdir,
                        "fichier": nc_file,
                        "lat_centre": lat_centre,
                        "lon_centre": lon_centre,
                        "sar_acquisition_time": time
                    })

                    ds.close()

                except Exception as e:
                    print(f"  ⚠️ Erreur avec {file_path} : {e}")


    # --- Conversion en DataFrame ---
    df = pd.DataFrame(infos)

    # --- Sauvegarde CSV ---
    output_csv = "/scale/user/mtannaou/alternance/excels/SARGEO_SAR_v00r00_09_janvier.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)


### create sargeo_sar_csv() ###
def add_sar_listing_path():
    sargeo_csv_path = "/scale/user/mtannaou/alternance/excels/SARGEO_SAR_v00r00_09_janvier.csv"
    listing_sar_path = "/scale/user/mtannaou/alternance/excels/listing_sar_09_janvier.csv"
    output_path = "/scale/user/mtannaou/alternance/excels/SARGEO_SAR_v00r00_09_janvier_v1.csv"
    listing_df = pd.read_csv(listing_sar_path)
    sargeo = pd.read_csv(sargeo_csv_path)
    #add inter column
    listing_df = listing_df.copy()
    listing_df["date_l2m"] = pd.to_datetime(
        listing_df["L2M path"].str.extract(r"(\d{8}t\d{6})")[0],
        format="%Y%m%dt%H%M%S",
        errors="coerce"
    )

    time_window = pd.Timedelta(minutes=3)

    for i, row in sargeo.iterrows():
        file_path = row["fichier"]
        #rcm1-sclnd-owi-cm-20250217t134001-20250217t134117-00003
        sargeo.loc[i, "sar_inter"] = file_path.split(".nc")[0][:-22]
        # search in listing_sar_path for matching sar_inter
        match = listing_df[listing_df["L2M path"].str.contains(sargeo.loc[i, "sar_inter"])]
        
        if len(match) > 0:
            sargeo.loc[i, "sar_path"] = match["L2M path"].values[0]
        else:
            # 2) Fallback : même satellite + delta temps <= 2 minutes

            # satellite = premier token avant "-"
            parts = sargeo.loc[i, "sar_inter"].split("-")
            if len(parts) < 5:
                continue

            satellite = parts[0]   # rcm1 / s1a / s1b etc.
            date_str = parts[4]    # 20250217t134001 (d'après ton format)

            date_ref = pd.to_datetime(date_str, format="%Y%m%dt%H%M%S", errors="coerce")
            if pd.isna(date_ref):
                continue

            match2 = listing_df[
                listing_df["L2M path"].str.contains(satellite, na=False)
                & listing_df["date_l2m"].between(date_ref - time_window, date_ref + time_window)
            ]

            if len(match2) > 0:
                sargeo.loc[i, "sar_path"] = match2["L2M path"].iloc[0]
    sargeo.to_csv(output_path, index=False)
    print(f"SARGEO with SAR path saved to {output_path}")
    
        









# ============================================================
# ================== METADATA EXTRACTION =====================
# ============================================================

def extract_sargeo_metadata(sargeo_root: str = SARGEO_ROOT):
    """
    Walks through SARGEO cyclone folders, loads IRWIN/WV NetCDF files,
    and extracts cyclone metadata.

    Returns:
        List of dictionaries that can be turned into a DataFrame.
    """

    metadata_records = []

    # List of cyclone folders (e.g. "al122024")
    cyclones = os.listdir(sargeo_root)

    for cyclone_id in cyclones:

        cyclone_path = os.path.join(sargeo_root, cyclone_id)
        if not os.path.isdir(cyclone_path):
            continue

        # Process only the IRWIN and WV subfolders
        for channel in ["IRWIN", "WV"]:
            channel_path = os.path.join(cyclone_path, channel)
            if not os.path.exists(channel_path):
                continue

            # List all NetCDF files
            nc_files = [f for f in os.listdir(channel_path) if f.endswith(".nc")]

            for nc_file in nc_files:
                file_path = os.path.join(channel_path, nc_file)

                try:
                    ds = xr.open_dataset(file_path)

                    # Extract metadata safely
                    lat_center = float(ds["storm_latitude"].values[4]) if "storm_latitude" in ds else None
                    lon_center = float(ds["storm_longitude"].values[4]) if "storm_longitude" in ds else None
                    acquisition_time = (
                        str(ds["sar_acquisition_time"].values) 
                        if "sar_acquisition_time" in ds 
                        else None
                    )

                    metadata_records.append({
                        "cyclone": cyclone_id,
                        "channel": channel,
                        "file": nc_file,
                        "lat_center": lat_center,
                        "lon_center": lon_center,
                        "acquisition_time": acquisition_time
                    })

                    ds.close()

                except Exception as err:
                    print(f"⚠️ Error reading {file_path}: {err}")

    return metadata_records


# ============================================================
# ======================== SAVE CSV ==========================
# ============================================================

def save_sargeo_csv(
    output_csv: str = "/scale/user/mtannaou/alternance/excels/sargeo.csv",
    sargeo_root: str = SARGEO_ROOT
):
    """
    Extracts metadata and writes it to a CSV file.
    """

    records = extract_sargeo_metadata(sargeo_root)
    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n📁 SARGEO metadata CSV saved at: {output_csv}")
    print(f"📌 Total NetCDF files indexed: {len(df)}")


# ============================================================
# ============================ RUN ============================
# ============================================================

if __name__ == "__main__":
    add_sar_listing_path()
