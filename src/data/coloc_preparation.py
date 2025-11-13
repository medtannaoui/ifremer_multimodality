# this script will create a data labelisation (IR - MW - SAR)

import os
import ast
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import pickle as pkl



# === Config ===
csv_path = "/scale/user/mtannaou/alternance/excels/colocates_sargeo_primed.csv"        #
sargeo_sar_csv_path = "/scale/user/mtannaou/alternance/excels/SARGEO_SAR.csv"
tcprimed_path = "/scale/project/ifremer-isi-jumeaunumerique/TC_PRIMED_DATASET/v01r01/final"   #tcprimed path
sargeo_path = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v01r02/cyclobs"    # sargeo path
sar_aeqd_path = "/scale/user/mtannaou/alternance/donnees_sar_changes" 


# the goal is to create a pkl files which contains 4 datasets (IR_tcprimed , MW_tcprimed , IR_sargeo , SAR_aeqd)
def create_coloc_pkl(out_put_path = "/scale/user/mtannaou/alternance/src/data_coloc_pkl"):
    sargeo_sar_df = pd.read_csv(sargeo_sar_csv_path)   # sargeo datarame
    df_coloc = pd.read_csv(csv_path)                   #dataframe of colocs
    for i , row in df_coloc[df_coloc["count"] != 0].reset_index().iterrows():

        datas = []
        
        cyclone_id = row["cyclone_id"]
        bassin = cyclone_id[:2].upper()
        num_cyclone = cyclone_id[2:4]
        year_cyclone = cyclone_id[4:]
        

        cyclone_primed_path = os.path.join(tcprimed_path,str(year_cyclone),bassin,str(num_cyclone))
        liste_file_nc = row["path_tcprimed"].split("]")[0].split("[")[-1].split(",")
        print("nombre de tcrpimed dans la liste",len(liste_file_nc))
        
        for nc_file in liste_file_nc :    #tcprimed files
            
            
            nc_file_path = os.path.join(cyclone_primed_path,nc_file[1:-1].split("'")[-1])
            print("--------######-------",nc_file_path)
            netcdf = nc.Dataset(nc_file_path)
            f = list(netcdf.groups.keys())
            ds_mw_s1_nc = xr.open_dataset(nc_file_path,group = f"{f[2]}/S1")
            ds_ir_tcprimed = xr.open_dataset(nc_file_path,group = f[-1])
            
            datas.append([ds_mw_s1_nc["x"],ds_mw_s1_nc["y"],ds_mw_s1_nc[list(ds_mw_s1_nc.data_vars.keys())[5]]])     #microwave data
            if "IRWIN" not in list(ds_ir_tcprimed.data_vars.keys()):
                
                continue

            datas.append(ds_ir_tcprimed["IRWIN"])  #infrared tcprimed data
            inter_sargeo_sar_path = row["path_sargeo"].split("ll")[0][:-7]
            
            sar_aeqd_path = sargeo_sar_df[sargeo_sar_df["fichier"].str.contains(inter_sargeo_sar_path)]["sar_xy"]
            

            #check of the file exist (if the center in sargeo exist
            if sargeo_sar_df[sargeo_sar_df["sar_xy"]==sar_aeqd_path.values[0]]["lon_centre"].values == 0 :
                continue

            sar_aeqd = xr.open_dataset(sar_aeqd_path.values[0])
            datas.append([sar_aeqd["x_sar"],sar_aeqd["y_sar"],sar_aeqd["owiWindSpeed"]])      #sar aeqd data
            

            sargeo_file = sargeo_sar_df[sargeo_sar_df["sar_xy"]==sar_aeqd_path.values[0]]["fichier"].values[0]
            
            sargeo_irwin_path = os.path.join(sargeo_path,cyclone_id,"IRWIN",sargeo_file)

            ds_iriwn_sargeo = xr.open_dataset(sargeo_irwin_path)["IRWIN"]

            ds_iriwn_sargeo = xr.open_dataset(sargeo_irwin_path)["IRWIN"]
            sargeo_ds = xr.open_dataset(sargeo_irwin_path)
            datas.append([ds_iriwn_sargeo.sel(t_rel=0).values,sargeo_ds["storm_latitude"].sel(t_rel=0).values,sargeo_ds["storm_latitude"].sel(t_rel=0).values])
        
        
            
            os.makedirs(out_put_path,exist_ok=True)
            with open(os.path.join(out_put_path,f"{str(row["date_sargeo"])}.pkl") , "wb") as f : 
                pkl.dump(datas,f)
            print(f"fichier {row['date_sargeo']} enregistré")
        


if __name__ == "__main__":
    create_coloc_pkl()