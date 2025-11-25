import torch
from torch.utils.data import Dataset
from pathlib import Path
import xarray as xr
import numpy as np
from loguru import logger
import pickle as pkl
from importlib import reload


import src.IR_to_SAR.data_preparation.data_preprocessing as dataprep
from src.visualisation.utils_colormap import CMAP
cmap_ir , cmap_sar = CMAP.cira_ir(), CMAP.cmap_sar()
reload(dataprep)



class PrepareDataSet():

    def __init__(self,pkl_file= "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_sar_ir_pkl/ir_sar_pairs_v1.pkl",
                 barycenter = "no",
                 size = 256,
                 norm = "z_score",
                 drop_nan_100 = True
                 ):
        print("Start preparation dataset : ......................")
        with open(pkl_file, "rb") as f :
            data_ir_sar = pkl.load(f)
        
        if barycenter == "yes":
                sars_recalib = dataprep.recenter_sar_around_barycenter(np.array(data_ir_sar)[:, 1, :, :])
                self.sar = np.array([item[0] for item in sars_recalib])
                self.dx_sar = [item[1] for item in sars_recalib]
                self.dy_sar = [item[2] for item in sars_recalib]
        else : 
                self.sar = np.array(data_ir_sar)[:, 1, :, :]
        
        self.ir = np.array(data_ir_sar)[:, 0, :, :]
        
        N,H,W = self.sar.shape    
        assert size <= H and size <= W, "Crop size must be <= original dimensions" 


        self.ir = self.ir[:, int(H//2 - size//2):int((H//2) + (size//2)), int((W//2) - (size//2)):int((W//2) + (size//2))]       
        self.sar = self.sar[:, int(H//2 - size//2):int((H//2) + (size//2)), int((W//2) - (size//2)):int((W//2) + (size//2))]  
            # self.mask_sar = dataprep.get_mask_of_nan_values(self.sar)
        indices_to_remove = [381, 129, 451, 680, 695, 772, 1070, 1285]  # deleted indexes

        self.ir  = np.delete(self.ir, indices_to_remove, axis=0) - 273.15
        self.sar = np.delete(self.sar, indices_to_remove, axis=0) * 1.94384

        
    
        if drop_nan_100 : 
            print("remove the couples where sar has more than 0.5 in 100km around the center")
            self.ir, self.sar = dataprep.remove_sar_nan(self.ir,self.sar)
        
        
        #create the mask of sar values 
        self.mask_sar = np.isfinite(self.sar).astype(np.float32)

        self.ir = np.nan_to_num(self.ir, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar = np.nan_to_num(self.sar, nan=0.0, posinf=0.0, neginf=0.0)
        print("NaN dans IR:", np.isnan(self.ir).sum())
        print("Inf dans IR:", np.isinf(self.ir).sum())
        print("Min/Max IR:", np.min(self.ir), np.max(self.ir))
        print(np.shape(np.array(self.ir)))

        
        
        print("NaN dans SAR:", np.isnan(self.sar).sum())
        print("Inf dans SAR:", np.isinf(self.sar).sum())
        print("Min/Max SAR:", np.min(self.sar), np.max(self.sar))
        print(np.shape(np.array(self.sar)))
        self.norm = norm 
        if self.norm == "z_score":
            self.ir, self.mean_val_ir, self.std_val_ir = dataprep.z_score(self.ir)
            self.sar , self.mean_val_sar, self.std_val_sar = dataprep.z_score(self.sar)
            self.max_val_ir, self.max_val_sar, self.min_val_ir, self.min_val_sar = None, None, None, None

        else : 
            self.ir, self.min_val_ir, self.max_val_ir = dataprep.min_max(self.ir)
            self.sar , self.min_val_sar, self.max_val_sar = dataprep.min_max(self.sar)
            self.mean_val_sar, self.mean_val_ir, self.std_val_sar, self.std_val_ir = None, None, None, None
            
        


        print(self.ir.shape,self.sar.shape)
        print("Data collected and normalized")
            
        
