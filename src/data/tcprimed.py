"""
This script scans the TC_PRIMED dataset and extracts cyclone-level metadata
into a CSV file.

For each cyclone, it retrieves:
    - Year
    - Basin (AL, EP, WP, IO, SH…)
    - Cyclone number
    - Environment file (_env_) used to define time range
    - Start and end timestamps extracted from filename

Output CSV columns:
    year, basin, cyclone_number, start_time, end_time, env_file
"""

import os
import re
import pandas as pd


# ============================================================
# ========================== PATHS ============================
# ============================================================

TCPRIMED_ROOT = "/scale/project/ifremer-isi-jumeaunumerique/TC_PRIMED_DATASET/v01r01/final"


# ============================================================
# ================== TC_PRIMED METADATA SCAN =================
# ============================================================

def extract_tcprimed_metadata(tcprimed_root: str = TCPRIMED_ROOT):
    """
    Scans the full TC_PRIMED directory tree and extracts cyclone metadata
    based on the "_env_" file, which contains start/end timestamps.

    Expected folder structure:
        TC_PRIMED/year/basin/cyclone_number/*.nc

    Returns:
        A list of metadata dictionaries.
    """

    metadata_records = []

    for year in sorted(os.listdir(tcprimed_root)):
        year_path = os.path.join(tcprimed_root, year)
        if not os.path.isdir(year_path):
            continue

        for basin in sorted(os.listdir(year_path)):
            basin_path = os.path.join(year_path, basin)
            if not os.path.isdir(basin_path):
                continue

            for cyclone_number in sorted(os.listdir(basin_path)):
                cyclone_path = os.path.join(basin_path, cyclone_number)
                if not os.path.isdir(cyclone_path):
                    continue

                # Find the environnement file (*_env_*.nc)
                env_files = [
                    f for f in os.listdir(cyclone_path)
                    if "_env_" in f and f.endswith(".nc")
                ]

                if not env_files:
                    continue

                env_file = env_files[0]  # always one environment file per cyclone

                # ------------------------------------------
                # Extract start and end timestamps from filename
                # Pattern: *_sYYYYMMDDHHMMSS_eYYYYMMDDHHMMSS.nc
                # ------------------------------------------
                match = re.search(r'_s(\d{14})_e(\d{14})', env_file)
                if not match:
                    print(f"⚠️ No valid timestamp pattern in {env_file}")
                    continue

                start_str, end_str = match.groups()

                # Format into ISO8601: YYYY-MM-DDTHH:MM:SS
                start_time = f"{start_str[:4]}-{start_str[4:6]}-{start_str[6:8]}T{start_str[8:10]}:{start_str[10:12]}:{start_str[12:]}"
                end_time   = f"{end_str[:4]}-{end_str[4:6]}-{end_str[6:8]}T{end_str[8:10]}:{end_str[10:12]}:{end_str[12:]}"

                metadata_records.append({
                    "year": year,
                    "basin": basin,
                    "cyclone_number": cyclone_number,
                    "start_time": start_time,
                    "end_time": end_time,
                    "env_file": env_file
                })

    return metadata_records


# ============================================================
# ======================== SAVE TO CSV ========================
# ============================================================

def save_tcprimed_csv(
    output_csv: str = "excels/tcprimed_data_env.csv",
    tcprimed_root: str = TCPRIMED_ROOT
):
    """
    Extracts metadata and saves it to a CSV file.
    """

    records = extract_tcprimed_metadata(tcprimed_root)
    df = pd.DataFrame(records)

    # Sort the table for readability
    df = df.sort_values(by=["year", "basin", "cyclone_number"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"📁 TC_PRIMED metadata CSV saved at: {output_csv}")
    print(f"📌 Cyclones indexed: {len(df)}")


# ============================================================
# ============================ RUN ============================
# ============================================================

if __name__ == "__main__":
    save_tcprimed_csv()
