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
                 ):

        print("🔹 Loading PKL data...")
        with open(pkl_file, "rb") as f:
            data = pkl.load(f)   # dictionary

        #  IRWIN is always included as the first channel ===
        final_channels = ["irwin"]
        if input_channels is not None:
            final_channels += input_channels   # append additional channels

        print(f"📎 Using input channels: {final_channels}")

        #  Extract metadata ===
        self.cyclone_ids = np.array(data["cyclone_id"])
        self.sar_time = np.array(data["sar_time"])
       

        #  Extract X channels ===
        #  image_channels : (N, H, W)
        #  feature_arrays : (N, F_i) qu'on concaténera en (N, F_total)
        image_channels = []
        feature_arrays = []
        feature_names = []   # for debug

        # all irwins (9)
        irwin_all = np.array(data["irwin"])    # (N, 9, H, W) par ex.
        N, _, H, W = irwin_all.shape

        # for i in [4]:
        #     irwin = irwin_all[:, i, :, :]      # (N, H, W)  # add multiple irs
        #     image_channels.append(irwin)

        image_channels.append(np.nanmean(irwin_all,axis=1)) #mean of th nine irs

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
            print("ℹ️ No features added to the blotelneck")

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
            self.X, self.sar = dataprep.remove_sar_nan(self.X, self.sar, radius_km=100, threshold=0.4)

        #  Create SAR valid pixel mask ===
        self.mask_sar = np.isfinite(self.sar).astype(np.float32)

        # Replace NaN & Inf ===
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar = np.nan_to_num(self.sar, nan=0.0, posinf=0.0, neginf=0.0)

        #  Normalization for SAR (always continuous) ===
        if norm == "z_score":
            self.sar, self.mean_sar, self.std_sar = dataprep.z_score(self.sar)
        else:
            self.sar, self.min_sar, self.max_sar = dataprep.min_max(self.sar)

        print("📌 SAR normalized.")

        #  Normalize only continuous input channels ===
        self.mean_X = {}
        self.std_X = {}
        self.min_X = {}
        self.max_X = {}

        C = self.X.shape[1]  # number of channels
        for c in range(C):
            channel_data = self.X[:, c, :, :]
            unique_vals = np.unique(channel_data)

            if len(unique_vals) > 20:   # Continuous channel → normalize
                if norm == "z_score":
                    norm_data, mean_val, std_val = dataprep.z_score(channel_data)
                    self.X[:, c, :, :] = norm_data
                    self.mean_X[c] = mean_val
                    self.std_X[c] = std_val
                else:  # Min-max normalization
                    norm_data, min_val, max_val = dataprep.min_max(channel_data)
                    self.X[:, c, :, :] = norm_data
                    self.min_X[c] = min_val
                    self.max_X[c] = max_val

               

            else:
                print(f"⏭️ Skipped channel {c} — only {len(unique_vals)} unique values (categorical)")

        
        print("🔍 Final X shape:", self.X.shape)  # (N, C, size, size)
        print("🔍 Final SAR shape:", np.expand_dims(self.sar, axis=1).shape)



        print(f"🎯 Dataset prepared successfully with {self.X.shape[1]} input channels.")

