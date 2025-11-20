# This script wil be used to test a simple U-NET for a regression simple (IR -> SAR)

import os
import sys
os.chdir("/scale/user/mtannaou/alternance")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

print("Python Path:", sys.path)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from loguru import logger
import lightning as L  # used for the callbacks
import dataclasses


print("---",os.getcwd())
from importlib import reload
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

import torch.nn.functional as F
import torchmetrics
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"🚀 Using device: {device}")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import lightning as L
import numpy as np
from tqdm import tqdm
import pickle as pkl


import src.IR_to_SAR.data_preprocessing as dataprep
from src.visualisation.utils_colormap import CMAP
cmap_ir , cmap_sar = CMAP.cira_ir(), CMAP.cmap_sar()
reload(dataprep)
from src.set_seed import set_seed
set_seed(0)

import src.IR_to_SAR.ML_IR_SAR.model as model_ir_sar
import src.IR_to_SAR.ML_IR_SAR.callbacks as callbacks
reload(model_ir_sar)
reload(callbacks)
from src.IR_to_SAR.ML_IR_SAR.model import create_model
from src.IR_to_SAR.ML_IR_SAR.callbacks import EarlyStopping,ModelCheckpoint,LogValidationSamples

import src.IR_to_SAR.ML_IR_SAR.config as config
reload(config)
from src.IR_to_SAR.ML_IR_SAR.config import IR_SAR_Config




# ============================================================
# ===============  DATASET  =================================
# ============================================================

class IRSARDataset(Dataset):
    """
    Dataset for IR → SAR prediction.
    IR shape:  (N, H_ir, W_ir)
    SAR shape: (N, H_ir, W_ir)
    """
    def __init__(self,test=False,size=256, norm = "z_score", barycenter = "no" ):
        self.norm = norm
        if not test : 
            with open("/scale/user/mtannaou/alternance/src/IR_to_SAR/data_sar_ir_pkl/ir_sar_pairs_v1.pkl","rb") as f :
                data_ir_sar = pkl.load(f)

            if barycenter == "yes":
                sars_recalib = dataprep.recenter_sar_around_barycenter(np.array(data_ir_sar)[:, 1, :, :])
                self.sar = np.array([item[0] for item in sars_recalib])
                self.dx_sar = [item[1] for item in sars_recalib]
                self.dy_sar = [item[2] for item in sars_recalib]

                print("sar data are recalibred around their barycenter")
            else : 
                self.sar = np.array(data_ir_sar)[:, 1, :size, :size]  


            
            self.ir = np.array(data_ir_sar)[:, 0, :size, :size]       
            self.sar = self.sar[:, :size, :size]
            # self.mask_sar = dataprep.get_mask_of_nan_values(self.sar)
            indices_to_remove = [381, 129, 451, 680, 695, 772, 1070, 1285]  # les indices à supprimer

            self.ir  = np.delete(self.ir, indices_to_remove, axis=0) - 273.15
            self.sar = np.delete(self.sar, indices_to_remove, axis=0) * 1.94384
            
            
        elif test : 
            self.ir = np.random.rand(10,16*3,16*3)
            self.sar = np.random.rand(10,16*3,16*3)
        
            
        print("NaN dans IR:", np.isnan(self.ir).sum())
        print("Inf dans IR:", np.isinf(self.ir).sum())
        print("Min/Max IR:", np.min(self.ir), np.max(self.ir))

        
        self.ir = np.nan_to_num(self.ir, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar = np.nan_to_num(self.sar, nan=0.0, posinf=0.0, neginf=0.0)
        print("NaN dans SAR:", np.isnan(self.sar).sum())
        print("Inf dans SAR:", np.isinf(self.sar).sum())
        print("Min/Max SAR:", np.min(self.sar), np.max(self.sar))

        if norm == "z_score":
            self.ir, self.mean_val_ir, self.std_val_ir = dataprep.z_score(self.ir)
            self.sar , self.mean_val_sar, self.std_val_sar = dataprep.z_score(self.sar)
            self.max_val_ir, self.max_val_sar, self.min_val_ir, self.min_val_sar = None, None, None, None

        else : 
            self.ir, self.min_val_ir, self.max_val_ir = dataprep.min_max(self.ir)
            self.sar , self.min_val_sar, self.max_val_sar = dataprep.min_max(self.sar)
            self.mean_val_sar, self.mean_val_ir, self.std_val_sar, self.std_val_ir = None, None, None, None
            

        print(self.ir.shape,self.sar.shape)
        print("Data collected and normalized")
        

    def __len__(self):    #number of observations
        return len(self.ir)

    def __getitem__(self, idx):
        ir  = self.ir[idx]   # (H,W)
        sar = self.sar[idx]  # (H,W)

        return ir , sar


class PairedDataset(Dataset):
    def __init__(self, ir_array, sar_array):
        self.ir = ir_array
        self.sar = sar_array

    def __len__(self):
        return len(self.ir)

    def __getitem__(self, idx):
        ir  = torch.tensor(self.ir[idx], dtype=torch.float32).unsqueeze(0)
        sar = torch.tensor(self.sar[idx], dtype=torch.float32).unsqueeze(0)
        return ir, sar



# =============================
# 🧠 2) Train / Validate functions
# =============================
def train_one_epoch(fabric, model, dataloader, optimizer, metrics):
    model.train()
    total_loss = 0
    metrics.reset()

    for ir, sar in tqdm(dataloader, desc="Training"):
        ir, sar = ir.to(fabric.device), sar.to(fabric.device)
        optimizer.zero_grad()

        # Forward
        pred = model(ir, timestep=0).sample   # (B,1,H,W)

        # Mask des pixels valides (ni NaN ni Inf)
        mask = torch.isfinite(sar)           # (B,1,H,W)

        loss = F.mse_loss(pred[mask], sar.nan_to_num()[mask])

        fabric.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        metrics.update(pred[mask], sar.nan_to_num()[mask])

    return total_loss / len(dataloader), metrics.compute()



def validate(fabric, model, dataloader, metrics):
    model.eval()
    total_loss = 0
    metrics.reset()

    with torch.no_grad():
        for ir, sar in tqdm(dataloader, desc="Validating"):
            ir, sar = ir.to(fabric.device), sar.to(fabric.device)

            pred = model(ir, timestep=0).sample

            mask = torch.isfinite(sar)

            loss = F.mse_loss(pred[mask], sar.nan_to_num()[mask])

            total_loss += loss.item()
            metrics.update(pred[mask], sar.nan_to_num()[mask])

    return total_loss / len(dataloader), metrics.compute()

# =============================
# 🚀 3) Main Training Lightning Fabric
# =============================

def main(cfg: IR_SAR_Config,test=False):
    logger.info(f"Starting Lightning training with config: {cfg}")

    # --- Dataset full ---
    
    full_data = IRSARDataset(test=test, size=cfg.img_size, norm=cfg.norm, barycenter=cfg.barycenter)
    ir_all, sar_all = full_data.ir, full_data.sar
    if cfg.barycenter == "yes" : 
        dx = full_data.dx_sar
        dy = full_data.dy_sar
    
    # --- Split ---
    dictio = dataprep.train_val_test_split(
        ir_all, sar_all,
        train_size=cfg.train_split,
        val_size=cfg.val_split,
        test_size=cfg.test_split
    )

    train_ds = PairedDataset(*dictio["train"])  #two attributes (ir and sar)
    val_ds   = PairedDataset(*dictio["val"])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # --- Fabric init with callbacks ---   #QUentin
    fabric = L.Fabric(
        accelerator=cfg.accelerator,
        devices= cfg.devices,
        strategy= "auto",
        callbacks=[
            EarlyStopping(patience=cfg.early_stop_patience, min_delta=cfg.early_stop_delta),
            ModelCheckpoint(cfg.save_dir),
            LogValidationSamples(
                base_dir=cfg.save_dir,
                num_samples=cfg.num_val_exemples,
                norm = full_data.norm,
                min_ir=full_data.min_val_ir,
                max_ir=full_data.max_val_ir,
                min_sar=full_data.min_val_sar,
                max_sar=full_data.max_val_sar,
                mean_sar = full_data.mean_val_sar,
                std_sar = full_data.std_val_sar,
                mean_ir = full_data.mean_val_ir,
                std_ir = full_data.std_val_ir,
                cmap_ir=cmap_ir,
                cmap_sar=cmap_sar,
                start_epoch=cfg.start_epoch,    
                 
            )
        ],
    )
    fabric.launch()

    # --- Metrics ---
    metrics = torchmetrics.MetricCollection({
        "mse": torchmetrics.MeanSquaredError(),
        "mae": torchmetrics.MeanAbsoluteError(),
    }).to(fabric.device)

    # --- Model & Optimizer & Scheduler ---
    model = create_model(image_size=cfg.img_size).to(fabric.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    # Prepare for Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # --- Training Loop ---
    for epoch in range(cfg.num_epochs):
        logger.info(f"===== Epoch {epoch+1}/{cfg.num_epochs} =====")
        
        train_loss, train_metrics = train_one_epoch(fabric, model, train_loader, optimizer, metrics)
        val_loss, val_metrics = validate(fabric, model, val_loader, metrics)

        scheduler.step()

        fabric.call(
            "on_validation_epoch_end",
            val_loss=val_loss,
            model=model,
            fabric=fabric
        )

        if (epoch + 1) >= cfg.start_epoch and (epoch + 1) % cfg.plot_interval == 0:
                fabric.call(
                "on_validation_plots",
                model=model,
                epoch=epoch,
                dataloader=val_loader,
                device=fabric.device
            )
                print("----- plots saved")

        fabric.print(
            f"📊 Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"LR={scheduler.get_last_lr()[0]:.6f}, "
            f"Train metrics={train_metrics}, Val metrics={val_metrics}"
        )

    logger.info("🎯 Training Complete!")


if __name__ == "__main__":
    from src.IR_to_SAR.ML_IR_SAR.config import IR_SAR_Config
    cfg = IR_SAR_Config.from_yaml("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/config.yaml")
    main(cfg,test=False)