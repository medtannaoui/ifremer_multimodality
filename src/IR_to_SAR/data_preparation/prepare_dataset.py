import torch
from torch.utils.data import Dataset
from pathlib import Path
import xarray as xr
import numpy as np
from loguru import logger
import pickle as pkl

from pathlib import Path
import os
from importlib import reload
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import re
from datetime import date, datetime, timedelta
import src.IR_to_SAR.data_preparation.data_preprocessing as dataprep
from src.visualisation.utils_colormap import CMAP
import src.IR_to_SAR.data_preparation.regrid_era5.regrid_era5 as regrid_colocs
cmap_ir , cmap_sar = CMAP.cira_ir(), CMAP.cmap_sar()
reload(dataprep)

era5_path = "/scale/user/mtannaou/alternance/src/extract_cyclones_era5/era5_single_levels"
janvier,mars,mai,juillet,aout,octobre,decembre = np.arange(1,32,1), np.arange(1,32,1), np.arange(1,32,1), np.arange(1,32,1), np.arange(1,32,1), np.arange(1,32,1), np.arange(1,32,1)
avril,juin,septembre,novembre = np.arange(1,31,1), np.arange(1,31,1), np.arange(1,31,1), np.arange(1,31,1)


def shift_ir_path(ir_path: str, idx: int, step_minutes: int = 30) -> str:
    m = re.search(r"(IR_)(\d{14})(\.nc)$", ir_path)
    if not m:
        raise ValueError(f"Format inattendu pour ir_path: {ir_path}")

    prefix, dt_str, suffix = m.group(1), m.group(2), m.group(3)
    dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    dt_shifted = dt + timedelta(minutes=idx * step_minutes)

    new_name = f"{prefix}{dt_shifted.strftime('%Y%m%d%H%M%S')}{suffix}"
    return ir_path[:m.start()] + new_name  # garde le même dossier

class PrepareDataSet():
    
    def __init__(self, pkl_file=None,
                  input_channels=None, 
                  barycenter="no", 
                  size=128, 
                  norm="z_score", 
                  drop_nan_100=True,
                  train_split=None,
                  val_split=None,
                  test_split=None,
                  augmentation=False,
                  target_dir = None,
                  input_data = "normal",
                  output_data = "sar",
                  conditional_model = None,
                  anggrek_test = False,
                  log_wind=False,
                  irwin_channels = 1,
                  regrid_ir = False,
                  ir_smoothing = False,
                  add_era5=False,cfg=None
                 ):
        self.train_split = train_split
        self.val_split=val_split
        self.test_split=test_split
        self.augmentation=augmentation
        self.target_dir=target_dir
        self.input_data = input_data
        self.output_data = output_data
        self.log_wind = log_wind
        self.irwin_channels= irwin_channels
        self.regrid_ir = regrid_ir
        self.ir_smoothing = ir_smoothing
        self.add_era5 = add_era5
        self.cfg=cfg
        

        print("🔹 Loading data from csv ...")
        # data = pd.read_csv("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/TCVA_matched_with_SARGEO_v3_split_by_year.csv")
        if not  conditional_model : 
            if not self.regrid_ir:
                data = pd.read_csv("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/TCVA_matched_with_SARGEO_v3_split_by_year.csv")[:]
            else : 
                data = pd.read_csv("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/tcva_matched_sargeo_4km_resolution.csv")[:]
        else :
            data = pd.read_csv("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/TCVA_matched_with_SARGEO_tcprimed.csv")
            data = data [~data["tcprimed_env_path"].isna()]
            
        # Anggrek Data 
        anggrek_csv = pd.read_csv("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/anggrek_coloc_sar_ir.csv")
        print("data before filtering", len(data))

        
        data = data[~data["split"].isna()]
        print("data after filtering",len(data))
        data = data.reset_index(drop=True)
        
        if True  : 
            if conditional_model:
                mag_cols = [f"shear_magnitude_{i}" for i in range(1, 9)]
                dir_cols = [f"shear_direction_{i}" for i in range(1, 9)]
                storm_cols = ["storm_speed","storm_speed_zonal_component","storm_speed_meridional_component"]
                cyclone_phase_space_thermal_wind = [f"cyclone_phase_space_thermal_wind_{i}" for i in range(1,3)]
                shear_cols = dir_cols+mag_cols
                train_shear = data[data["split"]=="train"][shear_cols].to_numpy(dtype=np.float32)
                train_shear = np.nan_to_num(train_shear, nan=0.0)

                shear_mean = train_shear.mean(axis=0)   # (16,)
                shear_std  = train_shear.std(axis=0)    # (16,)
            keys = ["cyclone_name","cyclone_id", "sar_time", "vmax",
                "analysis_vmax", "analysis_rmax",
                "analysis_center_quality_flag"]   #corilis
            good_rows, bad_rows = [], []
            good_rows_anggrek, bad_rows_anggrek = [], []
            irwin_train, irwin_val, irwin_test, irwin_anggrek= [], [], [], []
            sar_train, sar_val, sar_test, sar_anggrek = [], [], [], []
            infos_train, infos_val, infos_test, infos_anggrek = [], [], [], []
            era5_train, era5_val, era5_test , era5_anggrek = [], [], [], []

            N = len(data[data["split"]=="train"])
            pbar = tqdm(total=N, desc="Checking train files", unit="row")
            for i, row in data[data["split"]=="train"].iterrows():
                try:
                    with xr.open_dataset(row["sargeo_path"]) as sargeo:
                        if "IRWIN" not in sargeo:
                            raise KeyError("Missing IRWIN")
                            
                    with xr.open_dataset(row["sar_aeqd_path"]) as ds_aeqd:
                        if "owiWindSpeed" not in ds_aeqd:
                            raise KeyError("Missing owiWindSpeed")
                        
                        if self.add_era5:
                            sar_path = row["sar_aeqd_path"]
                            list_sar_path = [sar_path]
                            cyclone_id = row["cyclone_id"]
                            date = str(sar_path.split("/")[-1].split("-")[5])
                            year = date[0:4]; month = date[4:6];day = date[6:8]
                            hour = date[9:11]; minute = date[11:13]
                            year_path = os.path.join(era5_path, str(year))
                            fevrier = np.arange(1,29,1) if int(year)%4!=0 else np.arange(1,30,1)
                            months = [janvier,fevrier,mars,avril,mai,juin,juillet, aout, septembre, octobre, novembre, decembre]
                            ndays = 0
                            for i in range(int(month)-1):
                                ndays += len(months[i])
                            ndays += int(day)
                            ndays_str= "0"+str(ndays) if len(str(ndays)) < 3 else str(ndays)
                            dayera5_path = os.path.join(year_path, ndays_str)
                            nc_path = ""
                            for nc_file in os.listdir(dayera5_path):
                                if cyclone_id in nc_file:
                                    nc_path = os.path.join(dayera5_path, nc_file)
                                    break
                            reg_era5 = regrid_colocs.regrid_files_era5([nc_path], "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/regrid_era5/regridded_era5", 
                                           resolution_km=2, grid_size_km=300, list_sar_path=list_sar_path, index_hour=int(hour)-1+int(minute)//30)[0]
                            era5_train.append(reg_era5)
                            sar_train.append(ds_aeqd["owiWindSpeed"].values)
                            irwin_train.append(sargeo["IRWIN"].values)
                        else : 
                            sar_train.append(ds_aeqd["owiWindSpeed"].values)
                            irwin_train.append(sargeo["IRWIN"].values)

                    #add also environemental features
                    if conditional_model:
                        shear_vec = row[shear_cols].to_numpy(dtype=np.float32)
                        shear_vec = np.nan_to_num(shear_vec, nan=0.0)
                        shear_vec = (shear_vec - shear_mean) / shear_std
                        infos_train.append({
                                            **{k: row[k] for k in keys},
                                            "shear": shear_vec
                                        })
                    else : 
                        infos_train.append({k: row[k] for k in keys})

                    good_rows.append(i)

                except Exception as e:
                    print(e)
                    bad_rows.append(i)

                pbar.set_postfix(good=len(good_rows), bad=len(bad_rows))
                pbar.update(1)
            pbar.close()

            #validation set 
            N = len(data[data["split"]=="val"])
            pbar = tqdm(total=N, desc="Checking validation files", unit="row")
            for i, row in data[data["split"]=="val"].iterrows():
                try:
                    with xr.open_dataset(row["sargeo_path"]) as sargeo:
                        if "IRWIN" not in sargeo:
                            raise KeyError("Missing IRWIN")

                    with xr.open_dataset(row["sar_aeqd_path"]) as ds_aeqd:
                        if "owiWindSpeed" not in ds_aeqd:
                            raise KeyError("Missing owiWindSpeed")
                        
                        if self.add_era5:
                            sar_path = row["sar_aeqd_path"]
                            list_sar_path = [sar_path]
                            cyclone_id = row["cyclone_id"]
                            date = str(sar_path.split("/")[-1].split("-")[5])
                            year = date[0:4]; month = date[4:6];day = date[6:8]
                            hour = date[9:11]; minute = date[11:13]
                            year_path = os.path.join(era5_path, str(year))
                            fevrier = np.arange(1,29,1) if int(year)%4!=0 else np.arange(1,30,1)
                            months = [janvier,fevrier,mars,avril,mai,juin,juillet, aout, septembre, octobre, novembre, decembre]
                            ndays = 0
                            for i in range(int(month)-1):
                                ndays += len(months[i])
                            ndays += int(day)
                            ndays_str= "0"+str(ndays) if len(str(ndays)) < 3 else str(ndays)
                            dayera5_path = os.path.join(year_path, ndays_str)
                            nc_path = ""
                            for nc_file in os.listdir(dayera5_path):
                                if cyclone_id in nc_file:
                                    nc_path = os.path.join(dayera5_path, nc_file)
                                    break
                            reg_era5 = regrid_colocs.regrid_files_era5([nc_path], "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/regrid_era5/regridded_era5", 
                                           resolution_km=2, grid_size_km=300, list_sar_path=list_sar_path, index_hour=int(hour)-1+int(minute)//30)[0]
                            era5_val.append(reg_era5)
                            sar_val.append(ds_aeqd["owiWindSpeed"].values)
                            irwin_val.append(sargeo["IRWIN"].values)
                        else :
                            sar_val.append(ds_aeqd["owiWindSpeed"].values)
                            irwin_val.append(sargeo["IRWIN"].values)

                            
                    if conditional_model:
                        shear_vec = row[shear_cols].to_numpy(dtype=np.float32)
                        shear_vec = np.nan_to_num(shear_vec, nan=0.0)
                        shear_vec = (shear_vec - shear_mean) / shear_std
                        infos_val.append({
                                            **{k: row[k] for k in keys},
                                            "shear": shear_vec
                                        })
                    else : 
                        infos_val.append({k: row[k] for k in keys})

                    good_rows.append(i)

                except Exception as e:
                    bad_rows.append(i)

                pbar.set_postfix(good=len(good_rows), bad=len(bad_rows))
                pbar.update(1)
            pbar.close()

            #test set 
            N = len(data[data["split"]=="test"])
            pbar = tqdm(total=N, desc="Checking test files", unit="row")
            for i, row in data[data["split"]=="test"].iterrows():
                try:
                    with xr.open_dataset(row["sargeo_path"]) as sargeo:
                        if "IRWIN" not in sargeo:
                            raise KeyError("Missing IRWIN")
                        

                    with xr.open_dataset(row["sar_aeqd_path"]) as ds_aeqd:
                        if "owiWindSpeed" not in ds_aeqd:
                            raise KeyError("Missing owiWindSpeed")
                        

                        if self.add_era5:
                            sar_path = row["sar_aeqd_path"]
                            list_sar_path = [sar_path]
                            cyclone_id = row["cyclone_id"]
                            date = str(sar_path.split("/")[-1].split("-")[5])
                            year = date[0:4]; month = date[4:6];day = date[6:8]
                            hour = date[9:11]; minute = date[11:13]
                            year_path = os.path.join(era5_path, str(year))
                            fevrier = np.arange(1,29,1) if int(year)%4!=0 else np.arange(1,30,1)
                            months = [janvier,fevrier,mars,avril,mai,juin,juillet, aout, septembre, octobre, novembre, decembre]
                            ndays = 0
                            for i in range(int(month)-1):
                                ndays += len(months[i])
                            ndays += int(day)
                            ndays_str= "0"+str(ndays) if len(str(ndays)) < 3 else str(ndays)
                            dayera5_path = os.path.join(year_path, ndays_str)
                            nc_path = ""
                            for nc_file in os.listdir(dayera5_path):
                                if cyclone_id in nc_file:
                                    nc_path = os.path.join(dayera5_path, nc_file)
                                    break
                            reg_era5 = regrid_colocs.regrid_files_era5([nc_path], "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/regrid_era5/regridded_era5", 
                                           resolution_km=2, grid_size_km=300, list_sar_path=list_sar_path, index_hour=int(hour)-1+int(minute)//30)[0]
                            era5_test.append(reg_era5)
                            sar_test.append(ds_aeqd["owiWindSpeed"].values)
                            irwin_test.append(sargeo["IRWIN"].values)
                        else :
                            sar_test.append(ds_aeqd["owiWindSpeed"].values)
                            irwin_test.append(sargeo["IRWIN"].values)

                    if conditional_model:
                        shear_vec = row[shear_cols].to_numpy(dtype=np.float32)
                        shear_vec = np.nan_to_num(shear_vec, nan=0.0)
                        shear_vec = (shear_vec - shear_mean) / shear_std
                        infos_test.append({
                                            **{k: row[k] for k in keys},
                                            "shear": shear_vec
                                        })
                    else : 
                        infos_test.append({k: row[k] for k in keys})

                    good_rows.append(i)

                except Exception as e:
                    bad_rows.append(i)

                pbar.set_postfix(good=len(good_rows), bad=len(bad_rows))
                pbar.update(1)
            pbar.close()
            


            #anggrek cyclone
            if anggrek_test :
                N = len(anggrek_csv)
                indices = list(range(-(self.irwin_channels // 2), (self.irwin_channels // 2) + 1))
                pbar = tqdm(total=N, desc="Checking ANGGREK files", unit="row")
                for row_idx, row in anggrek_csv.iterrows():
                    ir_path = row["ir_path"]
                    paths = [shift_ir_path(ir_path, idx=i, step_minutes=30) for i in indices]

                    sample_imgs = []
                    ok = True

                    for path in paths:
                        try:
                            with xr.open_dataset(path) as ir_ds:
                                if "IR" not in ir_ds:
                                    ok = False
                                    break
                                arr = np.squeeze(ir_ds["IR"].values)
                                sample_imgs.append(arr)
                            good_rows_anggrek.append(row_idx)
                        except FileNotFoundError:
                            ok = False
                            bad_rows_anggrek.append(row_idx)
                            break
                    
                    # On n'ajoute le sample que si on a bien C canaux
                    if ok and len(sample_imgs) == len(indices):
                        # empile en (C, H, W)
                        irwin_anggrek.append(np.stack(sample_imgs, axis=0))

                        infos_anggrek.append({
                            "sid": row["sid"],
                            "date": row["date"],
                            "vmax": row["wind_speed (m/s)"],
                            "lat": row["lat"],
                            "lon": row["lon"],
                            "analysis_vmax_cyclobs": row["analysis_vmax_cyclobs"],
                            "vmax_cyclobs": row["vmax_cyclobs"],
                            "ibtracs_vmax": row["ibtracs_vmax"],
                            "satcon_vmax": row["satcon_vmax"],
                            "era5_vmax": row["era5_vmax"]
                        })
                        cyclone_id = anggrek_csv.iloc[row_idx]["sid"]
                        if self.add_era5:
                            
                            date = anggrek_csv.iloc[row_idx]["date"]
                            year = date[0:4]; month = date[5:7];day = date[8:10]
                            year_path = os.path.join(era5_path, str(year))
                            fevrier = np.arange(1,29,1) if int(year)%4!=0 else np.arange(1,30,1)
                            months = [janvier,fevrier,mars,avril,mai,juin,juillet, aout, septembre, octobre, novembre, decembre]
                            ndays = 0
                            for i in range(int(month)-1):
                                ndays += len(months[i])
                            ndays += int(day)
                            ndays_str= "0"+str(ndays) if len(str(ndays)) < 3 else str(ndays)
                            dayera5_path = os.path.join(year_path, ndays_str)
                            nc_path = ""
                            for nc_file in os.listdir(dayera5_path):
                                if cyclone_id in nc_file:
                                    nc_path = os.path.join(dayera5_path, nc_file)
                                    break
                            reg_era5 = regrid_colocs.regrid_files_era5([nc_path], "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/regrid_era5/regridded_era5", 
                                                                    resolution_km=2, grid_size_km=300, list_sar_path=list_sar_path, index_hour=int(hour)-1+int(minute)//30)
                            era5_anggrek.append(reg_era5)

                        

                

                    pbar.set_postfix(good=len(good_rows), bad=len(bad_rows))
                    pbar.update(1)
                # Final: (N, C, H, W)
                if self.add_era5:
                    era5_train = np.expand_dims(era5_train,axis=1)
                    era5_val = np.expand_dims(era5_val,axis=1)
                    era5_test = np.expand_dims(era5_test,axis=1)
                    era5_anggrek = np.array(era5_anggrek)    #np.expand_dims(era5_anggrek,axis=1)
                    print(np.array(irwin_anggrek).shape,np.array(era5_anggrek).shape)
                    print(np.array(irwin_train).shape, np.array(era5_train).shape)
                    print(np.array(irwin_val).shape,np.array(era5_val).shape)
                    print(np.array(irwin_test).shape, np.array(era5_test).shape)
                    h_era5,w_era5 = era5_anggrek.shape[-2],era5_anggrek.shape[-1]
                    h_anggrek,w_anggrek = np.array(irwin_anggrek).shape[-2],np.array(irwin_anggrek).shape[-1]
                    irwin_anggrek = np.array(irwin_anggrek)[:,:,h_anggrek//2-h_era5//2:h_anggrek//2+h_era5//2,h_anggrek//2-h_era5//2:h_anggrek//2+h_era5//2]
                    irwin_anggrek = np.concatenate([irwin_anggrek,era5_anggrek], axis=1)
                else : 
                    irwin_anggrek = np.array(irwin_anggrek) 
                    irwin_train = np.array(irwin_train)
                    irwin_val = np.array(irwin_val)
                    irwin_test = np.array(irwin_test)
                pbar.close()
                         
            #Filter dataframe to only good indices
            data = data.loc[good_rows].reset_index(drop=True)

            print(f"✅ Kept rows that open correctly: {len(data)}")
            print(f"❌ Dropped rows that failed: {len(bad_rows)}")

        self.infos_train = infos_train
        self.infos_val = infos_val
        self.infos_test = infos_test
        self.infos_anggrek = infos_anggrek if anggrek_test else None

        image_channels_train, image_channels_val, image_channels_test, image_channels_anggrek = [], [], [], []
    
        # all irwins (9)
        #add era5 to irwin 
        if self.add_era5:
            h_sargeo = np.array(irwin_train).shape[-1]
            irwin_train = np.array(irwin_train)[:,:,h_sargeo//2-h_era5//2:h_sargeo//2+h_era5//2,h_sargeo//2-w_era5//2:h_sargeo//2+w_era5//2]
            irwin_val= np.array(irwin_val)[:,:,h_sargeo//2-h_era5//2:h_sargeo//2+h_era5//2,h_sargeo//2-w_era5//2:h_sargeo//2+w_era5//2]
            irwin_test = np.array(irwin_test)[:,:,h_sargeo//2-h_era5//2:h_sargeo//2+h_era5//2,h_sargeo//2-w_era5//2:h_sargeo//2+w_era5//2]

            irwin_train = np.concatenate([irwin_train,era5_train],axis=1)   # (N, 9, H, W) par ex.
            
            
            irwin_val,irwin_test, irwin_anggrek = np.concatenate([irwin_val,era5_val],axis=1) ,np.concatenate([irwin_test,era5_test],axis=1) , np.array(irwin_anggrek) if anggrek_test else None
        self.sar_train, self.sar_val, self.sar_test = np.array(sar_train), np.array(sar_val), np.array(sar_test)

        print("train size :",len(sar_train))
        print("val_size : ",len(sar_val))
        print("Test size: ",len(sar_test))
        
        N, _, H, W = irwin_train.shape
        
        #train 
        if self.input_data == "all_channels":
            for i in [0,1,2,3,4,5,6,7,8]:
                image_channels_train.append(irwin_train[:, i, :, :]  )
                image_channels_val.append(irwin_val[:,i,:,:])
                image_channels_test.append(irwin_test[:,i,:,:])
                if anggrek_test :
                    image_channels_anggrek.append(irwin_anggrek)

        elif self.input_data == "normal":

            def transform_irwin_channels(x):
                """
                x shape: (N, 10, W, H)
                return shape: (N, 6, W, H)
                """
                c0 = x[:, 0:1, :, :]                       # garder canal 0
                c1 = x[:, 0:5, :, :].mean(axis=1, keepdims=True)   # mean(0,1,2,3,4)
                c2 = x[:, 2:5, :, :].mean(axis=1, keepdims=True)   # mean(2,3,4)
                c3 = x[:, 4:7, :, :].mean(axis=1, keepdims=True)   # mean(4,5,6)
                c4 = x[:, 4:9, :, :].mean(axis=1, keepdims=True)   # mean(4,5,6,7,8)
                c5 = x[:, 9:10, :, :]                      # garder canal 9

                return np.concatenate([c0, c1, c2, c3, c4, c5], axis=1)
            if not self.add_era5:
                irwin_train = irwin_train[:,4-self.irwin_channels//2:4+self.irwin_channels//2+1,:,:]
                irwin_val = irwin_val[:,4-self.irwin_channels//2:4+self.irwin_channels//2+1,:,:]
                irwin_test = irwin_test[:,4-self.irwin_channels//2:4+self.irwin_channels//2+1,:,:]
            
            image_channels_train.append(transform_irwin_channels(irwin_train) if self.ir_smoothing else irwin_train)
            image_channels_val.append(transform_irwin_channels(irwin_val) if self.ir_smoothing else irwin_val)
            image_channels_test.append(transform_irwin_channels(irwin_test) if self.ir_smoothing else irwin_test)

            if anggrek_test:
                image_channels_anggrek.append(transform_irwin_channels(irwin_anggrek) if self.ir_smoothing else irwin_anggrek)
        elif self.input_data == "normal+gradients":
            image_channels_train.append(irwin_train[:,4,:,:])
            image_channels_val.append(irwin_val[:,4,:,:])
            image_channels_test.append(irwin_test[:,4,:,:])
            if anggrek_test:
                image_channels_anggrek.append(irwin_anggrek) if anggrek_test else None

            image_channels_train.append(np.gradient(irwin_train[:,4,:,:])[0])
            image_channels_train.append(np.gradient(irwin_train[:,4,:,:])[1])

            image_channels_val.append(np.gradient(irwin_val[:,4,:,:])[0])
            image_channels_val.append(np.gradient(irwin_val[:,4,:,:])[1])

            image_channels_test.append(np.gradient(irwin_test[:,4,:,:])[0])
            image_channels_test.append(np.gradient(irwin_test[:,4,:,:])[1])
            if anggrek_test:
                image_channels_anggrek.append(np.gradient(irwin_anggrek)[0])
                image_channels_anggrek.append(np.gradient(irwin_anggrek)[1])

        elif self.input_data == "normal mean":
            image_channels_train.append(np.nanmean(irwin_train,axis=1)) 
            image_channels_val.append(np.nanmean(irwin_val,axis=1)) 
            image_channels_test.append(np.nanmean(irwin_test,axis=1)) 
            if anggrek_test:
                image_channels_anggrek.append(irwin_anggrek) 

        #stack the pictures list
        self.X_train = np.concatenate(image_channels_train, axis=1)  
        self.X_val = np.concatenate(image_channels_val, axis=1)
        self.X_test = np.concatenate(image_channels_test, axis=1)
        if anggrek_test:
            self.X_anggrek = np.array(image_channels_anggrek).squeeze()

 
        print("Start Reshaping Data ........")

        if self.regrid_ir:
            print("start regriding infrared data to 4km resolution : ")
            self.X_train = dataprep.regrid_batch_by_resolution(self.X_train,2,4)
            self.X_anggrek = dataprep.regrid_batch_by_resolution(self.X_anggrek,2,4)
            self.X_test = dataprep.regrid_batch_by_resolution(self.X_test,2,4)
            self.X_val = dataprep.regrid_batch_by_resolution(self.X_val,2,4)
            print("shape with 4 km resolution is ",self.X_train.shape)

        N, C, H, W = self.X_train.shape
        if anggrek_test:
            self.X_anggrek = self.X_anggrek
            
            N,C,h_anggrek,W_anggrek= self.X_anggrek.shape
        N,H_sar,W_sar = self.sar_train.shape
        self.X_train = self.X_train[:, :, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2] if not self.ir_smoothing else dataprep.build_irwin_channels(self.X_train[:, :, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2],9)
        self.X_val = self.X_val[:, :, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2] if not self.ir_smoothing else dataprep.build_irwin_channels(self.X_val[:, :, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2],9)
        self.X_test = self.X_test[:, :, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2] if not self.ir_smoothing else dataprep.build_irwin_channels(self.X_test[:, :, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2],9)
        if anggrek_test : 
            self.X_anggrek = self.X_anggrek[:, :, h_anggrek//2-size//2:h_anggrek//2+size//2, W_anggrek//2-size//2:W_anggrek//2+size//2] if not self.ir_smoothing else dataprep.build_irwin_channels(self.X_anggrek[:, :, h_anggrek//2-size//2:h_anggrek//2+size//2, W_anggrek//2-size//2:W_anggrek//2+size//2],9)


        self.sar_train = self.sar_train[:, H_sar//2-size//2:H_sar//2+size//2, W_sar//2-size//2:W_sar//2+size//2]
        self.sar_val = self.sar_val[:, H_sar//2-size//2:H_sar//2+size//2, W_sar//2-size//2:W_sar//2+size//2]
        self.sar_test = self.sar_test[:, H_sar//2-size//2:H_sar//2+size//2, W_sar//2-size//2:W_sar//2+size//2]
        # self.sar_anggrek = self.sar_anggrek[:, H_sar//2-size//2:H_sar//2+size//2, W_sar//2-size//2:W_sar//2+size//2]

        print("Final Shape of train Input is ",self.X_train.shape)
        print("Final Shape of train Output is ",self.sar_train.shape)
        print("Final Shape of Validation Input is ",self.X_val.shape)
        print("Final Shape of Validation Output is ",self.sar_val.shape)
        print("Final Shape of Test Input is ",self.X_test.shape)
        print("Final Shape of Test Output is ",self.sar_test.shape)




        #  Convert IR from Kelvin to Celsius ===
        n_ir_channels = self.X_train.shape[1] - 1 if self.add_era5 else self.X_train.shape[1]
        for c in range(n_ir_channels) : 
            self.X_train[:, c] = self.X_train[:, c] - 273.15  
            self.X_val[:,c] = self.X_val[:,c] - 273.15
            self.X_test[:,c] = self.X_test[:,c] - 273.15
            if anggrek_test:
                self.X_anggrek[:,c] = self.X_anggrek[:,c] - 273.15


        #  Create SAR valid pixel mask ===
        self.mask_train = np.isfinite(self.sar_train).astype(np.float32)
        self.mask_val = np.isfinite(self.sar_val).astype(np.float32)
        self.mask_test = np.isfinite(self.sar_test).astype(np.float32)
        # self.mask_anggrek = np.isfinite(self.sar_anggrek).astype(np.float32)


        # Replace NaN & Inf by 0 ===
        self.X_train = np.nan_to_num(self.X_train, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_val = np.nan_to_num(self.X_val, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_test = np.nan_to_num(self.X_test, nan=0.0, posinf=0.0, neginf=0.0)
        if anggrek_test:
            self.X_anggrek = np.nan_to_num(self.X_anggrek, nan=0.0, posinf=0.0, neginf=0.0)


        self.sar_train = np.nan_to_num(self.sar_train, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar_val = np.nan_to_num(self.sar_val, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar_test = np.nan_to_num(self.sar_test, nan=0.0, posinf=0.0, neginf=0.0)
        # self.sar_anggrek = np.nan_to_num(self.sar_anggrek, nan=0.0, posinf=0.0, neginf=0.0)

        if self.output_data == "aam":
            f_train = np.array([d["coriolis"] for d in self.infos_train], dtype=np.float32)
            f_val   = np.array([d["coriolis"] for d in self.infos_val], dtype=np.float32)
            f_test  = np.array([d["coriolis"] for d in self.infos_test], dtype=np.float32)

            if self.output_data == "aam":
                self.sar_train = dataprep.create_moment_sar(self.sar_train)
                self.sar_val = dataprep.create_moment_sar(self.sar_val)
                self.sar_test = dataprep.create_moment_sar(self.sar_test)
            # self.sar_anggrek = dataprep.create_moment_sar(self.sar_anggrek)

            # self.sar_anggrek = dataprep.create_moment_sar(self.sar_anggrek)


        if self.augmentation:
            
            print("Start Data Augmentation for train set : ----------")
            self.X_train, self.sar_train, self.mask_train, self.infos_train = dataprep.data_augmentation(
                    self.X_train, self.sar_train, self.mask_train, self.infos_train
                )
        print("New train size after augmentation is ",len(self.X_train))

        print("Plot Data distribution : --------------------")
        dataprep.plot_metric_scatter(
            true_values=[d["vmax"] for d in self.infos_train],
            pred_values=[d["analysis_vmax"] for d in self.infos_train],
            output_path=self.target_dir,
            file_name="analysis_vmax_and_vmax_comparaison_train",
            title="analysis vmax and vmax comparaison in the train set",
            xlabel="vmax (m\s)",
            ylabel="analysis_vmax (m\s)"
        )
        dataprep.plot_metric_scatter(
            true_values=[d["vmax"] for d in self.infos_val],
            pred_values=[d["analysis_vmax"] for d in self.infos_val],
            output_path=self.target_dir,
            file_name="analysis_vmax_and_vmax_comparaison_val",
            title="analysis vmax and vmax comparaison in the val set",
            xlabel="vmax (m\s)",
            ylabel="analysis_vmax (m\s)"
        )
        dataprep.plot_metric_scatter(
            true_values=[d["vmax"] for d in self.infos_test],
            pred_values=[d["analysis_vmax"] for d in self.infos_test],
            output_path=self.target_dir,
            file_name="analysis_vmax_and_vmax_comparaison_test",
            title="analysis vmax and vmax comparaison in the test set",
            xlabel="vmax (m\s)",
            ylabel="analysis_vmax (m\s)"
        )
        dataprep.plot_rmax_distribution(
            infos_train=self.infos_train,
            infos_val=self.infos_val,
            output_path=self.target_dir,
            file_name="analysis_rmax_distribution_train_vs_val.png"
        )
    



        print("Start Normalisation......")
        mean_x, std_x = [], []
        for c in range(self.X_train.shape[1]):
            self.X_train[:,c], mean, std = dataprep.z_score(self.X_train[:,c])
            mean_x.append(mean);std_x.append(std)
        self.mean_X,self.std_X = mean_x, std_x

        for c in range(self.X_train.shape[1]):
            self.X_val[:,c],_,_ = dataprep.z_score(self.X_val[:,c],mean_value=mean_x[c], std_value=std_x[c])
            self.X_test[:,c],_,_ = dataprep.z_score(self.X_test[:,c],mean_value=mean_x[c], std_value=std_x[c])
            if anggrek_test:
                self.X_anggrek[:,c],_,_ = dataprep.z_score(self.X_anggrek[:,c], mean_value=mean_x[c], std_value=std_x[c])

        
        
        if self.log_wind:
            self.sar_train = np.log(self.sar_train + 1e-10)
            self.sar_val = np.log(self.sar_val + 1e-10)
            self.sar_test = np.log(self.sar_test + 1e-10)
        if norm == "z_score":
            
            self.sar_train, self.mean_sar, self.std_sar = dataprep.z_score(self.sar_train)
            self.sar_val,_,_  = dataprep.z_score(self.sar_val,mean_value=self.mean_sar,std_value=self.std_sar)
            self.sar_test,_,_  = dataprep.z_score(self.sar_test,mean_value=self.mean_sar,std_value=self.std_sar)
            # self.sar_anggrek,_,_  = dataprep.z_score(self.sar_anggrek,mean_value=self.mean_sar,std_value=self.std_sar)
        elif norm == "annular":
            self.sar_train,stats = dataprep.annular_normalization(self.sar_train,bin_size=1,mask=None)
            self.mean_sar = stats["mean"]
            self.std_sar = stats["std"]
            self.sar_val,_ = dataprep.annular_normalization(self.sar_val,bin_size=1,mask=None,stats=stats)
            self.sar_test,_ = dataprep.annular_normalization(self.sar_test,mask=None,stats=stats)
            # self.sar_anggrek,_ = dataprep.annular_normalization(self.sar_anggrek,mask=None,stats=stats)

        out_stats_dir = os.path.join(self.target_dir, "stats_normalisation", "OUTPUT")
        os.makedirs(out_stats_dir, exist_ok=True)

        with open(os.path.join(out_stats_dir, "stats.pkl"), "wb") as f:
            pkl.dump(
                {
                    "normalisation_type": str(norm),
                    "std_sar": self.std_sar,
                    "mean_sar": self.mean_sar,
                    "mean_x": mean_x,
                    "std_x": std_x,
                },
                f
            )

        # ---------- ANGGREK data ----------
        if anggrek_test:
            anggrek_dir = os.path.join(
                self.target_dir, "stats_normalisation", "Anggrek_data_normalised"
            )
            os.makedirs(anggrek_dir, exist_ok=True)

            with open(os.path.join(anggrek_dir, "data.pkl"), "wb") as f:
                pkl.dump(
                    {
                        "IR_anggrek": self.X_anggrek,
                        "infos_anggrek": self.infos_anggrek,
                    },
                    f
            )
        print("SAR normalized.")

