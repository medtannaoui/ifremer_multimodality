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

SARGEO_ROOT = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v01r02/cyclobs"


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
    save_sargeo_csv()
