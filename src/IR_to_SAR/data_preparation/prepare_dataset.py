import os
import re
from datetime import datetime, timedelta
from importlib import reload

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import pickle as pkl

import src.IR_to_SAR.data_preparation.data_preprocessing as dataprep
import src.IR_to_SAR.data_preparation.regrid_era5.regrid_era5 as regrid_colocs
reload(dataprep)

era5_path = "/scale/user/mtannaou/alternance/src/extract_cyclones_era5/era5_single_levels"

# Tableaux représentant les jours de chaque mois (utilisés pour calculer le
# numéro du jour dans l'année, nécessaire pour retrouver le bon sous-dossier
# ERA5). Les mois à 31 jours et à 30 jours sont définis séparément ; le mois
# de février est recalculé dynamiquement (année bissextile ou non) au moment
# de son utilisation plus bas dans le code.
janvier, mars, mai, juillet, aout, octobre, decembre = (
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
)
avril, juin, septembre, novembre = (
    np.arange(1, 31, 1),
    np.arange(1, 31, 1),
    np.arange(1, 31, 1),
    np.arange(1, 31, 1),
)

class PrepareDataSet:
    def __init__(
        self,
        target_dir=None,
        cfg=None,
    ):
        self.augmentation = cfg.augmentation
        self.target_dir = target_dir
        self.output_data = cfg.output_data
        self.irwin_channels = cfg.irwin_channels
        self.add_era5 = cfg.add_era5
        self.cfg = cfg
        print("🔹 Loading data from csv ...")
        print("-------------------------------------------------------------")
        irwin_train, irwin_val, irwin_test, irwin_anggrek = [], [], [], []
        sar_train, sar_val, sar_test = [], [], []
        infos_train, infos_val, infos_test, infos_anggrek = [], [], [], []
        era5_train, era5_val, era5_test, era5_anggrek = [], [], [], []
        keys = [
                "cyclone_name",
                "cyclone_id",
                "sar_time",
                "analysis_vmax",
                "analysis_rmax",
                # "analysis_center_quality_flag",
            ]  # corilis
        ## prmeier cas : utilisation de la nouvelle base de données IRAR 
        if self.cfg.irar:
            data = pd.read_csv(
                "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/IRAR_L2_dataset.csv"
            )
            data = data[(data["valide"] == True) & (data["ir_regrided"] == True)]
            print("Len Data IRAR Valide :", len(data))

            irar = pd.read_csv("/scale/user/mtannaou/alternance/mnt/csvs_finaux/IRAR.csv")
            irar["date_irar"] = pd.to_datetime(irar["date_irar"])

            train_df = data[data["set"] == "train"][:10 if self.cfg.code_test else None]
            val_df = data[data["set"] == "val"][:10 if self.cfg.code_test else None]
            test_df = data[data["set"] == "test"][:10 if self.cfg.code_test else None]

            ## Train generating
            for df, name in zip([train_df, val_df, test_df], ["Train", "Val", "Test"]):
                for itr, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating {name} data Using IRAR Dataset : ..............."):
                    cyclone_id = row["cyclone_id"]
                    year = row["year"]
                    path_irar = row["path_irar"]
                    path_l2 = row["L2M path"]
                    date_irar = datetime.strptime(os.path.basename(path_irar).split("_s")[-1].split("_")[0], "%Y%m%d%H%M%S")

                    sequence_path_plus = []
                    sequence_path_moins = []
                    valid_sequence = True
                    index_par = range(1, 5)   # sargeo frequenc (range(1, 5))

                    for i in index_par:
                        date_i = date_irar + timedelta(minutes=i*30 if len(index_par)==4 else i*60)
                        date_j = date_irar - timedelta(minutes=i*30 if len(index_par)==4 else i*60)

                        sub_i = irar[(irar["date_irar"] == date_i) & (irar["cyclone_id"] == cyclone_id)]
                        sub_j = irar[(irar["date_irar"] == date_j) & (irar["cyclone_id"] == cyclone_id)]

                        if len(sub_i) == 0 or len(sub_j) == 0:
                            valid_sequence = False
                            break

                        sequence_path_plus.append(sub_i["path_nc"].values[0])
                        sequence_path_moins.append(sub_j["path_nc"].values[0])

                    if not valid_sequence:
                        continue

                    train_seq_paths = sequence_path_moins + [path_irar] + sequence_path_plus
                    irs = []
                    winds = []
                    valide = 0
                    for j, path in enumerate(train_seq_paths):
                            try : 
                                ds = xr.open_dataset(path)
                                # ds = dataprep.build_storm_centered_dataset(ds)

                                irs.append(ds["ir_aeqd"].values)   # (501,501) dxy = 2
                                if j == int(len(train_seq_paths)//2) :
                                    winds.append(ds["wind_aeqd"].values)  # (501,501) dxy = 2
                                
                                valide += 1
                            except Exception as e:
                                print(f"Error loading {path}: {e}")
                                break
                    
                    if valide == len(train_seq_paths):                       
                        if name == "Train":
                            irwin_train.append(np.stack(irs))
                            sar_train.append(np.stack(winds))
                            infos_train.append({k: row[k] for k in keys})

                        elif name == "Val":
                            irwin_val.append(np.stack(irs))
                            sar_val.append(np.stack(winds))
                            infos_val.append({k: row[k] for k in keys})

                        elif name == "Test":
                            irwin_test.append(np.stack(irs))
                            sar_test.append(np.stack(winds))
                            infos_test.append({k: row[k] for k in keys})


        ## Utilisation de sargeo avec une fenetre temporelle de 4h maximum avec 9 channels
        else :
            index_par = range(0, self.cfg.irwin_channels)
            if not cfg.conditional_model:
                
                data = pd.read_csv(
                        "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/"
                        "TCVA_matched_with_SARGEO_v3_split_by_year.csv"
                    )[: 50 if self.cfg.code_test else None]
                
            else:
                data = pd.read_csv(
                    "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/"
                    "TCVA_matched_with_SARGEO_tcprimed.csv"
                )
                data = data[~data["tcprimed_env_path"].isna()]

            # On ne garde que les lignes dont le split (train/val/test) est défini
            data = data[~data["split"].isna()]
            print("data after filtering", len(data))
            data = data.reset_index(drop=True)

            # --- Normalisation des features de cisaillement (shear) si modèle conditionnel ---
            if cfg.conditional_model:
                mag_cols = [f"shear_magnitude_{i}" for i in range(1, 9)]
                dir_cols = [f"shear_direction_{i}" for i in range(1, 9)]
                shear_cols = dir_cols + mag_cols

                train_shear = data[data["split"] == "train"][shear_cols].to_numpy(dtype=np.float32)
                train_shear = np.nan_to_num(train_shear, nan=0.0)

                shear_mean = train_shear.mean(axis=0)  # (16,)
                shear_std = train_shear.std(axis=0)    # (16,)
            

            ### Generating Data 
            for set_data in ["train", "val", "test"]:
                N = len(data[data["split"] == set_data])

                for i, row in tqdm(data[data["split"] == set_data].iterrows(), total=N, desc=f"Generating {set_data} data : ..............."):
                    try:
                        with xr.open_dataset(row["sargeo_path"]) as sargeo:
                            if "IRWIN" not in sargeo:
                                raise KeyError("Missing IRWIN")

                        with xr.open_dataset(row["sar_aeqd_path"]) as ds_aeqd:
                            if "owiWindSpeed" not in ds_aeqd:
                                raise KeyError("Missing owiWindSpeed")

                            # --- Récupération éventuelle des données ERA5 colocalisées ---
                            if self.add_era5:
                                sar_path = row["sar_aeqd_path"]
                                list_sar_path = [sar_path]
                                cyclone_id = row["cyclone_id"]
                                date = str(sar_path.split("/")[-1].split("-")[5])
                                year = date[0:4]
                                month = date[4:6]
                                day = date[6:8]
                                hour = date[9:11]
                                minute = date[11:13]
                                year_path = os.path.join(era5_path, str(year))
                                fevrier = np.arange(1, 29, 1) if int(year) % 4 != 0 else np.arange(1, 30, 1)
                                months = [
                                    janvier, fevrier, mars, avril, mai, juin,
                                    juillet, aout, septembre, octobre, novembre, decembre,
                                ]
                                ndays = 0
                                for i in range(int(month) - 1):
                                    ndays += len(months[i])
                                ndays += int(day)
                                ndays_str = "0" + str(ndays) if len(str(ndays)) < 3 else str(ndays)
                                dayera5_path = os.path.join(year_path, ndays_str)
                                nc_path = ""
                                for nc_file in os.listdir(dayera5_path):
                                    if cyclone_id in nc_file:
                                        nc_path = os.path.join(dayera5_path, nc_file)
                                        break
                                reg_era5 = regrid_colocs.regrid_files_era5(
                                    [nc_path],
                                    "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/"
                                    "regrid_era5/regridded_era5",
                                    resolution_km=2,
                                    grid_size_km=300,
                                    list_sar_path=list_sar_path,
                                    index_hour=int(hour) - 1 + int(minute) // 30,
                                )[0]

                                if set_data == "train":
                                    era5_train.append(reg_era5)
                                elif set_data == "val":
                                    era5_val.append(reg_era5)
                                elif set_data == "test":
                                    era5_test.append(reg_era5)
                            
                            if set_data == "train":
                                sar_train.append(ds_aeqd["owiWindSpeed"].values)
                                irwin_train.append(sargeo["IRWIN"].values)
                            elif set_data == "val":
                                sar_val.append(ds_aeqd["owiWindSpeed"].values)
                                irwin_val.append(sargeo["IRWIN"].values)
                            elif set_data == "test":
                                sar_test.append(ds_aeqd["owiWindSpeed"].values)
                                irwin_test.append(sargeo["IRWIN"].values)

                        # --- Ajout des features environnementales (shear) si modèle conditionnel ---
                        if cfg.conditional_model:
                            shear_vec = row[shear_cols].to_numpy(dtype=np.float32)
                            shear_vec = np.nan_to_num(shear_vec, nan=0.0)
                            shear_vec = (shear_vec - shear_mean) / shear_std
                            if set_data == "train":
                                infos_train.append({**{k: row[k] for k in keys}, "shear": shear_vec})
                            elif set_data == "val":
                                infos_val.append({**{k: row[k] for k in keys}, "shear": shear_vec})
                            elif set_data == "test":
                                infos_test.append({**{k: row[k] for k in keys}, "shear": shear_vec})
                        else:
                            if set_data == "train":
                                infos_train.append({k: row[k] for k in keys})
                            elif set_data == "val":
                                infos_val.append({k: row[k] for k in keys})
                            elif set_data == "test":
                                infos_test.append({k: row[k] for k in keys})

                    except Exception as e:
                        print(e)
                        continue
        if self.cfg.anggrek_test:
            # Données du cyclone "Anggrek" (jeu de test additionnel)
            anggrek_csv = pd.read_csv(
                "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/"
                "anggrek_coloc_sar_ir.csv"
            )[:10 if self.cfg.code_test else None]

            N = len(anggrek_csv)
            # Indices relatifs (-C/2 ... +C/2) des canaux IR temporels à charger
            indices = list(range(-(self.irwin_channels // 2), (self.irwin_channels // 2) + 1))
          
            for row_idx, row in tqdm(anggrek_csv.iterrows(), total=N, desc="Generating Anggrek data : ..............."):

                ir_path = row["ir_path"]
                paths = [
                        dataprep.shift_ir_path(ir_path, 
                                                idx=i, 
                                                step_minutes=30 if len(index_par) ==4 else 60) 
                                                for i in indices
                        ]

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
                    except FileNotFoundError:
                        ok = False
                        break

                # On n'ajoute le sample que si on a bien C canaux
                if ok and len(sample_imgs) == len(indices):
                    # empile en (C, H, W)
                    irwin_anggrek.append(np.stack(sample_imgs, axis=0))

                    infos_anggrek.append(
                        {
                            "sid": row["sid"],
                            "date": row["date"],
                            "vmax": row["wind_speed (m/s)"],
                            "lat": row["lat"],
                            "lon": row["lon"],
                            "analysis_vmax_cyclobs": row["analysis_vmax_cyclobs"],
                            "vmax_cyclobs": row["vmax_cyclobs"],
                            "ibtracs_vmax": row["ibtracs_vmax"],
                            "satcon_vmax": row["satcon_vmax"],
                            "era5_vmax": row["era5_vmax"],
                        }
                    )

                    cyclone_id = anggrek_csv.iloc[row_idx]["sid"]
                    if self.add_era5:
                        date = anggrek_csv.iloc[row_idx]["date"]
                        year = date[0:4]
                        month = date[5:7]
                        day = date[8:10]
                        year_path = os.path.join(era5_path, str(year))
                        fevrier = np.arange(1, 29, 1) if int(year) % 4 != 0 else np.arange(1, 30, 1)
                        months = [
                            janvier, fevrier, mars, avril, mai, juin,
                            juillet, aout, septembre, octobre, novembre, decembre,
                        ]
                        ndays = 0
                        for i in range(int(month) - 1):
                            ndays += len(months[i])
                        ndays += int(day)
                        ndays_str = "0" + str(ndays) if len(str(ndays)) < 3 else str(ndays)
                        dayera5_path = os.path.join(year_path, ndays_str)
                        nc_path = ""
                        for nc_file in os.listdir(dayera5_path):
                            if cyclone_id in nc_file:
                                nc_path = os.path.join(dayera5_path, nc_file)
                                break
                        reg_era5 = regrid_colocs.regrid_files_era5(
                            [nc_path],
                            "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/"
                            "regrid_era5/regridded_era5",
                            resolution_km=2,
                            grid_size_km=300,
                            list_sar_path=list_sar_path,
                            index_hour=int(hour) - 1 + int(minute) // 30,
                        )
                        era5_anggrek.append(reg_era5)
        
        self.X_train = np.stack(irwin_train); self.X_val = np.stack(irwin_val); self.X_test = np.stack(irwin_test)
        self.sar_train = np.stack(sar_train).squeeze(); self.sar_val = np.stack(sar_val).squeeze(); self.sar_test = np.stack(sar_test).squeeze()
        self.infos_train = np.array(infos_train) ; self.infos_val = np.array(infos_val) ; self.infos_test = np.array(infos_test)
        if self.cfg.anggrek_test:
            self.X_anggrek = np.array(irwin_anggrek) ; self.infos_anggrek = np.array(infos_anggrek)
        if self.add_era5:
            self.era5_train = np.array(era5_train) ; self.era5_val = np.array(era5_val) ; self.era5_test = np.array(era5_test)
            if self.cfg.anggrek_test:    
                self.era5_anggrek = np.array(era5_anggrek)
        
        print(f"irwin_train shape : {self.X_train.shape} ; sar_train shape : {self.sar_train.shape} ; infos_train shape : {self.infos_train.shape}")
        print(f"irwin_val shape : {self.X_val.shape} ; sar_val shape : {self.sar_val.shape} ; infos_val shape : {self.infos_val.shape}")
        print(f"irwin_test shape : {self.X_test.shape} ; sar_test shape : {self.sar_test.shape} ; infos_test shape : {self.infos_test.shape}")
        if self.cfg.anggrek_test:
            print(f"irwin_anggrek shape : {self.X_anggrek.shape} ; infos_anggrek shape : {self.infos_anggrek.shape}")
            if self.add_era5:
                print(f"era5_anggrek shape : {self.era5_anggrek.shape}")


        ## recadrage et centrage autour du cnetre et redimensionnement
        size = self.cfg.img_size
        N,C,H,W = self.X_train.shape
        self.X_train = self.X_train[:,:, W//2 - size//2 : W//2 + size//2, H//2 - size//2 : H//2 + size//2]
        self.X_val = self.X_val[:,:, W//2 - size//2 : W//2 + size//2, H//2 - size//2 : H//2 + size//2]
        self.X_test = self.X_test[:,:, W//2 - size//2 : W//2 + size//2, H//2 - size//2 : H//2 + size//2]
        if self.cfg.anggrek_test:
            N,C,H,W = self.X_anggrek.shape
            self.X_anggrek = self.X_anggrek[:,:, W//2 - size//2 : W//2 + size//2, H//2 - size//2 : H//2 + size//2]  
        self.sar_train = self.sar_train[:, W//2 - size//2 : W//2 + size//2, H//2 - size//2 : H//2 + size//2]
        self.sar_val = self.sar_val[:, W//2 - size//2 : W//2 + size//2, H//2 - size//2 : H//2 + size//2]
        self.sar_test = self.sar_test[:, W//2 - size//2 :  W//2 + size//2, H//2 - size//2 : H//2 + size//2]

        print(f"After Data resize : irwin_train shape : {self.X_train.shape} ; sar_train shape : {self.sar_train.shape}")

        ## conversion Kelvin -> Celsius
        self.X_train = self.X_train - 273.15 ; self.X_val = self.X_val - 273.15 ; self.X_test = self.X_test - 273.15
        if self.cfg.anggrek_test:
            self.X_anggrek = self.X_anggrek - 273.15
        

        ## masking NaN values for sar data
        self.mask_train = np.isfinite(self.sar_train).astype(np.float32)
        self.mask_val = np.isfinite(self.sar_val).astype(np.float32)
        self.mask_test = np.isfinite(self.sar_test).astype(np.float32)

        ## remplacement des nan par 0
        self.X_train = np.nan_to_num(self.X_train, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_val = np.nan_to_num(self.X_val, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_test = np.nan_to_num(self.X_test, nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg.anggrek_test:
            self.X_anggrek = np.nan_to_num(self.X_anggrek, nan=0.0, posinf=0.0, neginf=0.0)

        self.sar_train = np.nan_to_num(self.sar_train, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar_val = np.nan_to_num(self.sar_val, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar_test = np.nan_to_num(self.sar_test, nan=0.0, posinf=0.0, neginf=0.0)


        # Data Augmentation
        if self.cfg.augmentation : 
            print("Start Data Augmentation for train set : ----------")
            self.X_train, self.sar_train, self.mask_train, self.infos_train = dataprep.data_augmentation(
                                                                                            self.X_train, 
                                                                                            self.sar_train, 
                                                                                            self.mask_train, 
                                                                                            self.infos_train
                                                                                        )
            print("New Size after augmentation :",self.X_train.shape,self.sar_train.shape)
        

        ### Normalisation des entrées
        print("Start normalisation : .......................")

        mean_x, std_x = [], []
        for c in range(self.X_train.shape[1]):
            self.X_train[:, c], mean, std = dataprep.z_score(self.X_train[:, c])
            mean_x.append(mean)
            std_x.append(std)
        self.mean_X, self.std_X = mean_x, std_x

        for c in range(self.X_train.shape[1]):
            self.X_val[:, c], _, _ = dataprep.z_score(self.X_val[:, c], mean_value=mean_x[c], std_value=std_x[c])
            self.X_test[:, c], _, _ = dataprep.z_score(self.X_test[:, c], mean_value=mean_x[c], std_value=std_x[c])
            if self.cfg.anggrek_test:
                self.X_anggrek[:, c], _, _ = dataprep.z_score(
                    self.X_anggrek[:, c], mean_value=mean_x[c], std_value=std_x[c]
                )
        
        if self.cfg.norm == "z_score":
            self.sar_train, self.mean_sar, self.std_sar = dataprep.z_score(self.sar_train)
            self.sar_val, _, _ = dataprep.z_score(self.sar_val, mean_value=self.mean_sar, std_value=self.std_sar)
            self.sar_test, _, _ = dataprep.z_score(self.sar_test, mean_value=self.mean_sar, std_value=self.std_sar)

        elif self.cfg.norm == "annular":
            self.sar_train, stats = dataprep.annular_normalization(self.sar_train, bin_size=1, mask=None)
            self.mean_sar = stats["mean"]
            self.std_sar = stats["std"]
            self.sar_val, _ = dataprep.annular_normalization(self.sar_val, bin_size=1, mask=None, stats=stats)
            self.sar_test, _ = dataprep.annular_normalization(self.sar_test, mask=None, stats=stats)

        ## sauvgarde des données et les stats de normlisation 
        data_saved = os.path.join(self.target_dir, "Test_data_stats")
        os.makedirs(data_saved, exist_ok=True)

        with open(os.path.join(data_saved, "Tets_data_stats.pkl"), "wb") as f:
            pkl.dump(
                {
                    "normalisation_type": str(self.cfg.norm),
                    "std_sar": self.std_sar,
                    "mean_sar": self.mean_sar,
                    "mean_x": mean_x,
                    "std_x": std_x,
                    "test_set_irwin": self.X_test,
                    "test_set_sar": self.sar_test,
                    "anggrek": self.X_anggrek
                },
                f,
            )
        
        print("Data Preparation finished.")
