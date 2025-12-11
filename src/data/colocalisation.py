"""
This script performs temporal colocation between:
    - SARGEO IRWIN data
    - TC_PRIMED overpass data

It generates a CSV listing all matched timestamps and file paths:
    cyclone_id, date_sargeo, date_tcprimed, path_sargeo, path_tcprimed, count
"""

import os
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# ========================== PATHS ============================
# ============================================================

TCPRIMED_DATA_PATH = "/scale/project/ifremer-isi-jumeaunumerique/TC_PRIMED_DATASET/v01r01/final"
SARGEO_DATA_PATH   = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v00r00/cyclobs"


# ============================================================
# ================== TIME MATCHING FUNCTION ==================
# ============================================================

def is_within_time_window(date_sar: str, date_primed: str, delta_minutes: int = 10) -> bool:
    """
    Checks whether a SARGEO timestamp is within ±delta minutes of a TC_PRIMED timestamp.

    Inputs:
        date_sar     (str)  : format "YYYYMMDDTHHMMSS"
        date_primed  (str)  : format "YYYY-MM-DDTHH:MM:SS.000"
        delta_minutes (int) : allowed time difference in minutes

    Returns:
        bool : True if timestamps are temporally colocated
    """

    fmt_sar = "%Y%m%dT%H%M%S"
    fmt_primed = "%Y-%m-%dT%H:%M:%S.%f"

    sar_dt = datetime.strptime(date_sar, fmt_sar)
    primed_dt = datetime.strptime(date_primed, fmt_primed)

    delta = timedelta(minutes=delta_minutes)
    return (primed_dt - delta) <= sar_dt <= (primed_dt + delta)


# ============================================================
# ====================== LOAD TCPRIMED INFO ==================
# ============================================================

def get_tcprimed_overview():
    """
    Scans TC_PRIMED directory structure and returns cyclone metadata:

    Output DataFrame columns:
        year, basin, cyclone_number, files_count, cyclone_id
    """

    records = {"year": [], "basin": [], "cyclone_number": [], "files_count": [], "cyclone_id": []}

    for year in os.listdir(TCPRIMED_DATA_PATH):
        year_path = os.path.join(TCPRIMED_DATA_PATH, year)
        if not os.path.isdir(year_path):
            continue

        for basin in os.listdir(year_path):
            basin_path = os.path.join(year_path, basin)

            for num in os.listdir(basin_path):
                cyclone_path = os.path.join(basin_path, num)
                files_count = len(os.listdir(cyclone_path))

                records["year"].append(year)
                records["basin"].append(basin)
                records["cyclone_number"].append(num)
                records["files_count"].append(files_count)
                records["cyclone_id"].append(basin + num + year)

    return pd.DataFrame(records)


def time_diff_seconds(date_sar: str, date_primed: str) -> float:
    """Returns the absolute difference between timestamps in seconds."""
    fmt_sar = "%Y%m%dT%H%M%S"
    fmt_primed = "%Y-%m-%dT%H:%M:%S.%f"
    sar_dt = datetime.strptime(date_sar, fmt_sar)
    primed_dt = datetime.strptime(date_primed, fmt_primed)
    return abs((sar_dt - primed_dt).total_seconds())


# ============================================================
# ======================== COLOCATION ========================
# ============================================================

def colocate_sar_tcprimed(
    sargeo_csv_path: str = "excels/SARGEO_SAR_V4.csv",
    tcprimed_csv_path: str = "excels/TCPrimed_overpass.csv",
    time_window_minutes: int = 90
):
    """
    Performs temporal colocation between SARGEO IRWIN products and TC_PRIMED overpasses.

    Inputs:
        sargeo_csv_path     : CSV with SARGEO IRWIN metadata
        tcprimed_csv_path   : CSV with TC_PRIMED overpass metadata
        time_window_minutes : matching threshold in minutes

    Returns:
        DataFrame with columns:
            cyclone_id, count, date_sargeo, date_tcprimed,
            path_sargeo, path_tcprimed
    """

    results = {}
    data_sargeo = pd.read_csv(sargeo_csv_path)
    data_primed = pd.read_csv(tcprimed_csv_path, sep=";")

    # Build TC_PRIMED cyclone ID
    data_primed["cyclone_id"] = (
        data_primed["bassin"] +
        data_primed["num_cyclone"].astype(str) +
        data_primed["annee"].astype(str)
    )
    
    for cyclone_id in data_sargeo["cyclone"].unique():

        tc_subset = data_primed[data_primed["cyclone_id"] == cyclone_id.upper()]
        sargeo_ir_path = os.path.join(SARGEO_DATA_PATH, cyclone_id, "IRWIN")

        results[cyclone_id] = []
        # print(os.listdir(sargeo_ir_path))
        for filename in os.listdir(sargeo_ir_path):

            full_sargeo_path = os.path.join(sargeo_ir_path, filename)
            
            sar_date_raw = filename.split("-")[4]     # Extract raw date "YYYYMMDDTHHMMSS"

            matches = []
            matched_tc_paths = []
            matched_tc_dates = []
            matched_diffs_times = []

            for _, row in tc_subset.iterrows():
                if is_within_time_window(sar_date_raw, row["date"], time_window_minutes):

                    matches.append(row["date"])
                    matched_tc_paths.append(os.path.basename(row["path"]))
                    matched_tc_dates.append(row["date"])

                    # compute difference in seconds
                    diff_sec = time_diff_seconds(sar_date_raw, row["date"])
                    matched_diffs_times.append(diff_sec)
                  

                    # print(f"Matched TC_PRIMED date → {row['date']}")

            iso_sar_date = datetime.strptime(sar_date_raw, "%Y%m%dT%H%M%S").strftime("%Y-%m-%dT%H:%M:%S.000")

            results[cyclone_id].append({
                "cyclone_id": cyclone_id,
                "count": len(matches),
                "date_sargeo": iso_sar_date,
                "date_tcprimed": matched_tc_dates,
                "diff_time": matched_diffs_times,
                "path_sargeo": os.path.basename(full_sargeo_path),
                "path_tcprimed": matched_tc_paths
            })

    # Flatten into a DataFrame
    flat_rows = []
    for cyclone_id, entries in results.items():
        flat_rows.extend(entries)

    df = pd.DataFrame(flat_rows)
    return df


# ============================================================
# ========================= SAVE CSV =========================
# ============================================================

def save_colocation_csv(
    output_path: str = "excels/colocates_sargeo_primed_v1.csv",
    sargeo_csv_path: str = "excels/SARGEO_cyclones.csv",
    tcprimed_csv_path: str = "excels/TCPrimed_overpass.csv",
    delta: int = 90
):
    """
    Runs the colocation function and saves the result to CSV.
    """
    df = colocate_sar_tcprimed(
        sargeo_csv_path=sargeo_csv_path,
        tcprimed_csv_path=tcprimed_csv_path,
        time_window_minutes=delta
    )
    df.to_csv(output_path, index=False)
    print(f"✓ Colocation CSV saved: {output_path}")


# ============================================================
# ============================ RUN ============================
# ============================================================

if __name__ == "__main__":
    delta = 90
    save_colocation_csv(output_path= f"excels/colocates_sargeo_primed_{delta}min_v1.csv", delta = delta)
