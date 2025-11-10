# This script will be used to create a csv format for TCPrimed dataset

import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd
import re


run = True

main_path = "/scale/project/ifremer-isi-jumeaunumerique/TC_PRIMED_DATASET/v01r01/final"   #tcprimed path
years = [i for i in range(1987, 2025)]    # list of years
bassins = ['AL', 'CP','EP', 'IO', 'SH', 'WP']   #list of bassins


def create_tcprimed_csv():
    
    records = []

    # iterate on the years list
    for annee in sorted(os.listdir(main_path)):
        annee_path = os.path.join(main_path, annee)
        if not os.path.isdir(annee_path):
            continue

        # iiterate on the bassins list
        for bassin in sorted(os.listdir(annee_path)):
            bassin_path = os.path.join(annee_path, bassin)
            if not os.path.isdir(bassin_path):
                continue

            # iterate the cyclone numbers for this bassins in this year
            for cyclone_num in sorted(os.listdir(bassin_path)):
                cyclone_path = os.path.join(bassin_path, cyclone_num)
                if not os.path.isdir(cyclone_path):
                    continue

                # take the env file
                env_files = [f for f in os.listdir(cyclone_path) if "_env_" in f and f.endswith(".nc")]
                if not env_files:
                    continue  # no env file found

                # we take the first env file in the list(always there is one file environenetal)
                env_file = env_files[0]

                # extract the start and the end of the env file date
                match = re.search(r'_s(\d{14})_e(\d{14})', env_file)
                if match:
                    start_str, end_str = match.groups()
                    # convert to lsisble format
                    start_fmt = f"{start_str[:4]}-{start_str[4:6]}-{start_str[6:8]}T{start_str[8:10]}:{start_str[10:12]}:{start_str[12:]}"
                    end_fmt = f"{end_str[:4]}-{end_str[4:6]}-{end_str[6:8]}T{end_str[8:10]}:{end_str[10:12]}:{end_str[12:]}"

                    records.append({
                        "annee": annee,
                        "bassin": bassin,
                        "cyclone_numero": cyclone_num,
                        "date_debut": start_fmt,
                        "date_fin": end_fmt,
                        "fichier": env_file
                    })
    return records
    


def main(out_put_dir = "excels/tcprimed_data_env.csv"):
    #save the dataframe on to csv file
    # create a pandas dataframe
    df = pd.DataFrame(create_tcprimed_csv())

    # sort the observations
    df = df.sort_values(by=["annee", "bassin", "cyclone_numero"]).reset_index(drop=True)

    # save on csv file
    df.to_csv(out_put_dir, index=False)

    print(f"file saved in : {out_put_dir}")




if run :
    main()
