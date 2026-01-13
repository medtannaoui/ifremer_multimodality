import torch
from torch.utils.data import Dataset
from pathlib import Path
import xarray as xr
import numpy as np
from loguru import logger
import pickle as pkl
from importlib import reload
import matplotlib.pyplot as plt

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
        

        print("🔹 Loading PKL data...")
        with open(pkl_file, "rb") as f:
            data = pkl.load(f)   # dictionary
#         mask = [x <= 1.4 for x in data["analysis_center_quality_flag"]]

#         data = {
#             key: [value[i] for i in range(len(value)) if mask[i]]
#             for key, value in data.items()
# }
        #  IRWIN is always included as the first channel ===
        final_channels = ["irwin"]
        if input_channels is not None:
            final_channels += input_channels   # append additional channels

        print(f"📎 Using input channels: {final_channels}")

        #  Extract metadata ===
        keys = ["cyclone_id", "sar_time", "vmax", 
        "analysis_vmax", "analysis_rmax", 
        "analysis_center_quality_flag"]

        self.infos = [
            {k: data[k][i] for k in keys}
            for i in range(len(data["cyclone_id"]))
        ]
                        

        #  Extract X channels ===
       
        image_channels = []
        feature_arrays = []
        feature_names = []   # for debug

        # all irwins (9)
        irwin_all = np.array(data["irwin"])    # (N, 9, H, W) par ex.
        N, _, H, W = irwin_all.shape

        # for i in [0,1,2,3,4,5,6,7,8]:
        #     irwin = irwin_all[:, i, :, :]      # (N, H, W)  # add multiple irs
        #     image_channels.append(irwin)
        image_channels.append(irwin_all[:,4,:,:])
        image_channels.append(np.gradient(irwin_all[:,4,:,:])[0])
        image_channels.append(np.gradient(irwin_all[:,4,:,:])[1])

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
        self.sar = np.array(data["owiwindspeed"])

    
        # Center crop 
        N, C, H, W = self.X.shape
        self.X = self.X[:, :, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2]
        self.sar = self.sar[:, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2]

        #  Convert IR from Kelvin to Celsius ===
        self.X[:, 0] = self.X[:, 0] - 273.15   # always channel 0 = irwin

        # Optional: remove SAR samples with too many NaNs ===
        if drop_nan_100:
            self.X, self.sar, self.infos = dataprep.remove_sar_nan(self.X, self.sar, radius_km=128, threshold=0.7, infos=self.infos)

        #  Create SAR valid pixel mask ===
        self.mask_sar = np.isfinite(self.sar).astype(np.float32)

        # Replace NaN & Inf ===
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar = np.nan_to_num(self.sar, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar = dataprep.create_moment_sar(self.sar)

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

