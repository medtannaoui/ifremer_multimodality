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
from src.IR_to_SAR.data_preparation.distribution_data_visualisation import plot_ir, plot_sar,plot_ir_hist,plot_sar_hist




class PrepareDataSet():
    
    def __init__(self, pkl_file="/scale/user/mtannaou/alternance/src/IR_to_SAR/data_sar_ir_pkl/ir_sar_pairs_eye_dictv1.pkl",
                  input_channels=None, 
                  barycenter="no", 
                  size=256, 
                  norm="z_score", 
                  drop_nan_100=True):

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
        # self.eye_cenetr_lat = np.array(data["eye_center_lat"])
        # self.eye_center_lon = np.array(data["eye_center_lon"])
        # self.storm_center_lat = np.array(data["storm_center_lat"])
        # self.storm_center_lon = np.array(data["storm_center_lon"])

        #  Extract X channels ===
        X_list = []

        # Always add IRWIN first
        irwin = np.array(data["irwin"])  # (N,H,W)
        

        N,H,W = irwin.shape
        X_list.append(irwin)

        # Process additional input channels
        
        for var in input_channels:
            arr = np.array(data[var])

            if arr.ndim == 1:
                arr = np.tile(arr[:, None, None], (1, H, W))
                X_list.append(arr)

            elif arr.ndim == 2:
                for c in range(arr.shape[1]):
                    expanded = np.tile(arr[:, c][:, None, None], (1, H, W))
                    X_list.append(expanded)

            elif arr.ndim == 3:
                # 🎯 Case (N, H’, 1) → Flatten into H' scalar channels
                if arr.shape[2] == 1:
                    H_prime = arr.shape[1]
                    print(f"Expanding {var}: shape (N,{H_prime},1) → {H_prime} scalar channels")

                    arr_2d = arr.reshape(arr.shape[0], H_prime)  # (N, H')
                    for c in range(H_prime):
                        expanded = np.tile(arr_2d[:, c][:, None, None], (1, H, W))
                        X_list.append(expanded)

                    

                elif arr.shape[1:] == (H, W):
                    X_list.append(arr)  # Normal (N,H,W) image channel

                else:
                    arr = arr.reshape(arr.shape[0],arr.shape[1]*arr.shape[2])
                    for c in range(arr.shape[1]):
                        expanded = np.tile(arr[:, c][:, None, None], (1, H, W))
                    X_list.append(expanded)
                    
                    

            else:
                raise ValueError(f"Unsupported number of dimensions for {var}: {arr.shape}")
            
            print(f"{var}-------{arr.shape}")



        # Final stack → (N,C,H,W)
   
        self.X = np.stack(X_list, axis=1)

        #  Extract SAR windspeed as target ===
        self.sar = np.array(data["owiwindspeed"])

        # Center crop ===
        N, C, H, W = self.X.shape
        self.X = self.X[:, :, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2]
        self.sar = self.sar[:, H//2-size//2:H//2+size//2, W//2-size//2:W//2+size//2]

        #  Convert IR from Kelvin to Celsius ===
        self.X[:, 0] = self.X[:, 0] - 273.15   # always channel 0 = irwin

        # Optional: remove SAR samples with too many NaNs ===
        if drop_nan_100:
            self.X, self.sar = dataprep.remove_sar_nan(self.X, self.sar, radius_km=100, threshold=0.1)

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

