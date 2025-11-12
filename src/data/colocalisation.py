# This script will be used to collocate SAR data with TCPRIMED data
import pandas as pd
import numpy as np
import os
import xarray as xr
from matplotlib import pyplot as plt
from datetime import datetime,timedelta
import importlib



run = True


tcprimed_data_path = "/scale/project/ifremer-isi-jumeaunumerique/TC_PRIMED_DATASET/v01r01/final"
sargeo_data_path = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v01r02/cyclobs"


def is_within_deltamin(date1_str, date2_str, delta=10):
    """
    Checks whether date1 falls within the interval [date2 - delta minutes, date2 + delta minutes].

    Parameters:
        date1_str (str): first date, e.g. '2019-09-20T22:19:52.000'
        date2_str (str): second date, e.g. '2019-09-20T22:25:00.000'
        delta (int or float): time window in minutes

    Returns:
        bool: True if date1 is within the interval, otherwise False
    
    """
    fmt1 = "%Y%m%dT%H%M%S"
    fmt2 = "%Y-%m-%dT%H:%M:%S.%f"
    
    # CConvert to datetime
    d1 = datetime.strptime(date1_str, fmt1)
    d2 = datetime.strptime(date2_str, fmt2)
    
    # Creation of the ±delta-minute interval
    delta = timedelta(minutes=delta)
    
    return (d2 - delta) <= d1 <= (d2 + delta)



#function which return tcprimed infos 
def get_tcrpimed_infos():

    df_tcprimed = {"year":[],"bassin":[],"cyclone_number":[],"files_count":[]}
    for year in os.listdir(tcprimed_data_path):
        year_path = os.path.join(tcprimed_data_path,year)
    for bassin in os.listdir(year_path):
        bassin_path = os.path.join(year_path,bassin)
        for num in os.listdir(bassin_path):
            files_count = len(os.listdir(os.path.join(bassin_path,num)))
            df_tcprimed["year"].append(year)
            df_tcprimed["bassin"].append(bassin)
            df_tcprimed["cyclone_number"].append(num)
            df_tcprimed["files_count"].append(files_count)
    
    tc_primed_infos = pd.DataFrame(df_tcprimed,columns=list(df_tcprimed.keys()))
    
    #add cyclone_id column ( like al022025) for facilate the comparison with sargeo data
    for i,row in tc_primed_infos.iterrows(): 
        tc_primed_infos.loc[i,"cyclone_id"] = row["bassin"]+row["cyclone_number"]+row["year"]

    return  tc_primed_infos


#colocate SARGEO and TCPrimed
def colocate(sargeo_path="excels/SARGEO_cyclones.csv", tcprimed_overpass_path = "excels/TCPrimed_overpass.csv", delta = 30):
    rslt = {}
    canal = "IRWIN"

    #sargeo data
    sargeo_data = pd.read_csv(sargeo_path)
    #tcprimed data overpass
    tcprimed_data = pd.read_csv(tcprimed_overpass_path,sep=";")
    
    for i,row in tcprimed_data.iterrows() : 
        tcprimed_data.loc[i,"cyclone_id"] = row["bassin"]+str(row["num_cyclone"])+str(row["annee"])

    for cyc_exemple in sargeo_data["cyclone"].unique():

        extrait_primed = tcprimed_data[tcprimed_data["cyclone_id"] == cyc_exemple.upper()]
        path_irwin = os.path.join(sargeo_data_path, cyc_exemple, canal)

        rslt[cyc_exemple] = []

        for file in os.listdir(path_irwin):
            path_sargeo = os.path.join(path_irwin, file)
            date = str(file.split("-")[4])

            compt_int = 0
            tc_paths = []   
            date_tcprimed = []
            for _, row in extrait_primed.iterrows():
                if is_within_deltamin(date, row["date"], delta):
                    compt_int += 1
                    tc_paths.append(os.path.basename(row["path"]))
                    date_tcprimed.append(row["date"])

            rslt[cyc_exemple].append({
                # "date": date,
                "count": compt_int,
                "path_sargeo": os.path.basename(path_sargeo),
                "path_tcprimed": tc_paths,
                "date_sargeo":datetime.strptime(date,"%Y%m%dT%H%M%S").strftime("%H-%m-%d-T%H:%M:%S.000"),
                "date_tcprimed": date_tcprimed
            })

    # ✅ Conversion en DataFrame
    rows = []
    for cyclone_id, entries in rslt.items():
        for e in entries:
            rows.append({
                "cyclone_id": cyclone_id,
                # "date": e["date"],
                "count": e["count"],
                "date_sargeo":e["date_sargeo"],
                "date_tcprimed": e["date_tcprimed"],
                "path_sargeo": e["path_sargeo"],
                "path_tcprimed": e["path_tcprimed"]
            })

    return  pd.DataFrame(rows)


#save on csv file

def save_colocate_data(sargeo_path="excels/SARGEO_cyclones.csv", tcprimed_overpass_path = "excels/TCPrimed_overpass.csv", delta = 30, output_dir = "excels/colocates_sargeo_primed.csv"):
    data = colocate()
    data.to_csv(output_dir)
    print("file saved on ",output_dir)


if run : 
    # print(colocate().head(10))
    save_colocate_data()



