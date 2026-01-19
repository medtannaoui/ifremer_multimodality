import torch
from torch.utils.data import Dataset
from pathlib import Path
import xarray as xr
import numpy as np
from loguru import logger
import pickle as pkl
import os
from importlib import reload
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import src.IR_to_SAR.data_preparation.data_preprocessing as dataprep
from src.visualisation.utils_colormap import CMAP
cmap_ir , cmap_sar = CMAP.cira_ir(), CMAP.cmap_sar()
reload(dataprep)




class PrepareDataSet():
    
    def __init__(self, pkl_file=None,
                  input_channels=None, 
                  barycenter="no", 
                  size=256, 
                  norm="z_score", 
                  drop_nan_100=True,
                  train_split=None,val_split=None,test_split=None,augmentation=False,target_dir = None
                 ):
        self.train_split = train_split
        self.val_split=val_split
        self.test_split=test_split
        self.augmentation=augmentation
        self.target_dir=target_dir
        

        print("🔹 Loading data...")
        # with open(pkl_file, "rb") as f:
        #     data = pkl.load(f)   # dictionary
        print("🔹 Loading data...")
        data = pd.read_csv("/scale/user/mtannaou/alternance/TCVA_matched_with_SARGEO_df.csv")
        print("data before filtering", len(data))

        data = data[~data["sar_aeqd_path"].isnull()].reset_index(drop=True)
        print("data after filtering (non-null sar_aeqd_path)", len(data))

        data =data[data["nan_ratio_within_100km"] <= 0.5]
        print("data after filter with nan_ratio within radius of 100km",len(data))

        data =data[data["analysis_rmax"] <= 180000]
        print("data size after remove SAR with analysis_rmax bigger than 180Km")


        final_channels = ["irwin"]
        print(f"📎 Using input channels: {final_channels}")

        # ---- Keep only rows that open well ----
        
        
        N = len(data)
        if os.path.exists("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/pairs_from_csv.pkl"):
            with open("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/pairs_from_csv.pkl","rb") as f: 
                data_pkl = pkl.load(f)
            sar_all  = np.array(data_pkl["owiWindSpeed"])
            irwin_all = np.array(data_pkl["irwin"])
        else : 
            good_rows, bad_rows = [], []
            irwin_all = []
            sar_all = []
            pbar = tqdm(total=N, desc="Checking files", unit="row")
            for i, row in data.iterrows():
                try:
                    with xr.open_dataset(row["sargeo_path"]) as sargeo:
                        if "IRWIN" not in sargeo:
                            raise KeyError("Missing IRWIN")
                        irwin_all.append(sargeo["IRWIN"].values)

                    with xr.open_dataset(row["sar_aeqd_path"]) as ds_aeqd:
                        if "owiWindSpeed" not in ds_aeqd:
                            raise KeyError("Missing owiWindSpeed")
                        sar_all.append(ds_aeqd["owiWindSpeed"].values)

                    good_rows.append(i)

                except Exception as e:
                    bad_rows.append(i)

                pbar.set_postfix(good=len(good_rows), bad=len(bad_rows))
                pbar.update(1)

            pbar.close()
            with open("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/pairs_from_csv.pkl","wb") as f:
                pkl.dump({"owiWindSpeed":np.array(sar_all),"irwin":np.array(irwin_all)},f)
                         
            # Filter dataframe to only good indices
            data = data.loc[good_rows].reset_index(drop=True)

            print(f"✅ Kept rows that open correctly: {len(data)}")
            print(f"❌ Dropped rows that failed: {len(bad_rows)}")



        irwin_all = np.array(irwin_all)
        sar_all = np.array(sar_all)

        # ---- Now proceed safely ----

        keys = ["cyclone_id", "sar_time", "vmax",
                "analysis_vmax", "analysis_rmax",
                "analysis_center_quality_flag"]

        self.infos = data[keys].to_dict(orient="records")

        image_channels = []
        feature_arrays = []
        feature_names = [] 
        

       
    
        # all irwins (9)
        irwin_all = np.array(irwin_all)    # (N, 9, H, W) par ex.
        N, _, H, W = irwin_all.shape

        # for i in [0,1,2,3,4,5,6,7,8]:
        #     irwin = irwin_all[:, i, :, :]      # (N, H, W)  # add multiple irs
        #     image_channels.append(irwin)
        image_channels.append(irwin_all[:,4,:,:])
        # image_channels.append(np.gradient(irwin_all[:,4,:,:])[0])
        # image_channels.append(np.gradient(irwin_all[:,4,:,:])[1])

        # image_channels.append(np.nanmean(irwin_all,axis=1)) #mean of th nine irs

        if input_channels is not None:
            for var in input_channels:
                arr = np.array(data[var])   # shape variable (N, ...) 

                # if a feature has already a shape like irwin (N,H,W) we add it to the features list
                if arr.ndim == 3 and arr.shape[1:] == (H, W):
                    print(f"{var}: traité comme IMAGE (N,H,W) = {arr.shape}")
                    image_channels.append(arr)

                # flatt the last dimensions so we have (N,F)
                else:
                    # 
                    arr_flat = arr.reshape(N, -1)   # (N, F_var)
                    feature_arrays.append(arr_flat)
                    feature_names.append(var)
                    print(f"{var}: Features with shape = {arr_flat.shape}")

        # 3) stack the pictures list
        self.X = np.stack(image_channels, axis=1)   # (N, C, H, W)

        # concatenate the faetures primed
        if len(feature_arrays) > 0:
            self.X_features = np.concatenate(feature_arrays, axis=1)  # (N, F_total)
            self.feature_names = feature_names
            print("🔍 Final features shape:", self.X_features.shape)
            print("📎 Feature names:", self.feature_names)
        else:
            self.X_features = None
            self.feature_names = []
            print("ℹ️ No features added to the bottleneck")

        #  Extract SAR windspeed as target
        self.sar = np.array(sar_all)

    
        # Center crop 
        print(self.sar.shape)
        print(self.X.shape)
        N, C, H, W = self.X.shape
        N,H_sar,W_sar = self.sar.shape
        self.X = self.X[:, :, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2]
        self.sar = self.sar[:, H_sar//2-size//2:H_sar//2+size//2, W_sar//2-size//2:W_sar//2+size//2]
        print(self.sar.shape)
        print(self.X.shape)

        #  Convert IR from Kelvin to Celsius ===
        self.X[:, 0] = self.X[:, 0] - 273.15   # always channel 0 = irwin

        #  Create SAR valid pixel mask ===
        self.mask_sar = np.isfinite(self.sar).astype(np.float32)

        # Replace NaN & Inf ===
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar = np.nan_to_num(self.sar, nan=0.0, posinf=0.0, neginf=0.0)
        # self.sar = dataprep.create_moment_sar(self.sar)

        # --- Split ---
        self.dictio = dataprep.train_val_test_split(
            np.array(self.X), np.array(self.sar),
            train_size=self.train_split,
            val_size=self.val_split,
            test_size=self.test_split,
            augmentation=self.augmentation,
            mask_sar = self.mask_sar,
            infos = self.infos,
            target_dir=self.target_dir
        )


        #"substract mean radial profil "
        self.radial_profil = None
        # self.sar_train,self.radial_profil = dataprep.subtract_radial_mean(self.dictio["train"][1],bin_size=1)
        # self.sar_val, _ = dataprep.subtract_radial_mean(self.dictio["val"][1],radial_profil=self.radial_profil)
        # self.sar_test, _ = dataprep.subtract_radial_mean(self.dictio["test"][1],radial_profil=self.radial_profil)
        self.sar_train = self.dictio["train"][1]
        self.sar_val = self.dictio["val"][1]
        self.sar_test = self.dictio["test"][1]
        #  Normalization for SAR (always continuous) ===

        if norm == "z_score":
            self.sar_train, self.mean_sar, self.std_sar = dataprep.z_score(self.sar_train)
            self.sar_val,_,_  = dataprep.z_score(self.sar_val,mean_value=self.mean_sar,std_value=self.std_sar)
            self.sar_test,_,_  = dataprep.z_score(self.sar_test,mean_value=self.mean_sar,std_value=self.std_sar)

        elif norm == "annular":
            self.sar_train,stats = dataprep.annular_normalization(self.sar_train,bin_size=1,mask=None)
            self.mean_sar = stats["mean"]
            self.std_sar = stats["std"]
            self.sar_val,_ = dataprep.annular_normalization(self.sar_val,bin_size=1,mask=None,stats=stats)
            self.sar_test,_ = dataprep.annular_normalization(self.sar_test,mask=None,stats=stats)
                    
        else:
            self.sar, self.min_sar, self.max_sar = dataprep.min_max(self.sar)

        print("📌 SAR normalized.")

        #  Normalize only continuous input channels ===
        self.mean_X = {}
        self.std_X = {}
        self.min_X = {}
        self.max_X = {}
        X_train = self.dictio["train"][0]   # shape (N, C, H, W)
        X_val   = self.dictio["val"][0]
        X_test  = self.dictio["test"][0]

        # Initialisation
        self.X_train = X_train.copy()
        self.X_val   = X_val.copy()
        self.X_test  = X_test.copy()

        C = self.X.shape[1]  # number of channels
        for c in range(C):
            channel_data = self.dictio["train"][0][:,c,:,:]
            unique_vals = np.unique(channel_data)

            if np.nanstd(channel_data) > 1e-6:   # Continuous channel → normalize
                if True:
                    norm_data, mean_val, std_val = dataprep.z_score(channel_data)
                    self.X_train[:, c, :, :] = norm_data
                    self.mean_X[c] = mean_val
                    self.std_X[c] = std_val
                # elif norm =="minmax":  # Min-max normalization
                #     norm_data, min_val, max_val = dataprep.min_max(channel_data)
                #     self.X_train[:, c, :, :] = norm_data
                #     self.min_X[c] = min_val
                #     self.max_X[c] = max_val
                # elif norm == "annular":
                #     norm_data,stats = dataprep.annular_normalization(channel_data,bin_size=1)
                #     self.X_train[:,c,:,:] = norm_data
                #     self.mean_X[c] = stats["mean"]
                #     self.std_X[c] = stats["std"]
        for set in ["val","test"]:
            for c in range(C):
                channel_data = self.dictio[set][0][:,c,:,:]
                unique_vals = np.unique(channel_data)

                if np.nanstd(channel_data) > 1e-8:   # Continuous channel → normalize
                    if True:
                        norm_data, mean_val, std_val = dataprep.z_score(channel_data,mean_value=self.mean_X[c],std_value=self.std_X[c])
                        if set =="val":
                            self.X_val[:, c, :, :] = norm_data
                        else : 
                            self.X_test[:,c,:,:]=norm_data
                        
                    # elif norm =="minmax":  # Min-max normalization
                    #     norm_data, min_val, max_val = dataprep.min_max(channel_data)
                    #     self.X_train[:, c, :, :] = norm_data
                    #     self.min_X[c] = min_val
                    #     self.max_X[c] = max_val
                    # elif norm == "annular":
                    #     norm_data,stats = dataprep.annular_normalization(channel_data,bin_size=1,mean_value=self.mean_X[c],std_value=self.std_X[c])
                    #     if set =="val":
                    #         self.X_val[:, c, :, :] = norm_data
                    #     else : 
                    #         self.X_test[:,c,:,:]=norm_data
                        
                

               

                else:
                    print(f"⏭️ Skipped channel {c} — only {len(unique_vals)} unique values (categorical)")

        
        print("🔍 Final X shape:", self.X.shape)  # (N, C, size, size)
        print("🔍 Final SAR shape:", np.expand_dims(self.sar, axis=1).shape)



        print(f"🎯 Dataset prepared successfully with {self.X.shape[1]} input channels.")

