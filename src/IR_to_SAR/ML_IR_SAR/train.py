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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



print("---",os.getcwd())
from importlib import reload
import torch
torch.set_float32_matmul_precision('high')
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional import structural_similarity_index_measure as ssim
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"🚀 Using device: {device}")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import lightning as L
import numpy as np
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


import src.IR_to_SAR.data_preparation.data_preprocessing as dataprep
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
import src.IR_to_SAR.data_preparation.prepare_dataset as prep_dataset
reload(prep_dataset)




# ============================================================
# ===============  DATASET  =================================
# ============================================================

class IRSARDataset(Dataset):
    """
    Dataset for IR → SAR prediction.
    IR shape:  (N, H_ir, W_ir)
    SAR shape: (N, H_ir, W_ir)
    """
    def __init__(self,test=False,size=256, norm = "z_score", barycenter = "no" ,augmentation = False, drop_nan_100 = True):
        self.norm = norm
        
        dataset = prep_dataset.PrepareDataSet(size=size, norm= norm, barycenter= barycenter, drop_nan_100=drop_nan_100)
        self.ir = dataset.ir
        self.sar = dataset.sar   
        self.dataset = dataset     
        print("Data preparation finished")
        print(self.ir.shape, self.sar.shape)

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
        pred = model(ir, timestep=0).sample  # (B,1,H,W)

        # Mask des pixels valides
        mask = torch.isfinite(sar)

        sar_valid = sar.nan_to_num()
        pred_valid = pred

        # --- Loss MSE ---
        loss_mse = F.mse_loss(pred_valid[mask], sar_valid[mask])

        # --- Loss SSIM (on calcule sur toute l’image, sans mask) ---
        loss_ssim = 1 - ssim(pred_valid, sar_valid)   # (1 - SSIM) car SSIM = similarité

        # --- Combine loss ---
        loss = 0 * loss_mse + 1 * loss_ssim

      
        fabric.backward(loss)
        fabric.clip_gradients(model,optimizer, max_norm=1.0)   #gradient clipping
        optimizer.step()

        total_loss += loss.item()
        metrics.update(pred_valid[mask], sar_valid[mask])

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
            sar_valid = sar.nan_to_num()
            pred_valid = pred

            # Loss MSE
            loss_mse = F.mse_loss(pred_valid[mask], sar_valid[mask])

            # Loss SSIM (1 - SSIM car SSIM = Similarité)
            loss_ssim = 1 - ssim(pred_valid, sar_valid)

            # Combine
            loss = 0 * loss_mse + 1 * loss_ssim

            total_loss += loss.item()
            metrics.update(pred_valid[mask], sar_valid[mask])

    return total_loss / len(dataloader), metrics.compute()

# =============================
# 🚀 3) Main Training Lightning Fabric
# =============================

def main(cfg: IR_SAR_Config,test=False):
    logger.info(f"Starting Lightning training with config: {cfg}")

    # --- Dataset full ---
    
    full_data = IRSARDataset(test=test, size=cfg.img_size, norm=cfg.norm, barycenter=cfg.barycenter, drop_nan_100=cfg.drop_nan_sar)
    ir_all, sar_all = full_data.dataset.ir, full_data.dataset.sar
    if cfg.barycenter == "yes" : 
        dx = full_data.dataset.dx_sar
        dy = full_data.dataset.dy_sar
    
    # --- Split ---
    dictio = dataprep.train_val_test_split(
        np.array(ir_all), np.array(sar_all),
        train_size=cfg.train_split,
        val_size=cfg.val_split,
        test_size=cfg.test_split,
        augmentation=cfg.augmentation,
        mask_sar = full_data.dataset.mask_sar
    )

    train_ds = PairedDataset(*dictio["train"])  #two attributes (ir and sar)
    val_ds   = PairedDataset(*dictio["val"])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # --- Fabric init with callbacks ---   #QUentin
    base_dir = Path(cfg.save_dir)

    i = 1
    while (base_dir / f"train_ir_sar_{i}").exists():
        i += 1

    target_dir = base_dir / f"train_ir_sar_{i}"
    fabric = L.Fabric(
        accelerator=cfg.accelerator,
        devices= cfg.devices,
        strategy= "auto",
        callbacks=[
            EarlyStopping(patience=cfg.early_stop_patience, min_delta=cfg.early_stop_delta),
            ModelCheckpoint(cfg.save_dir, target_dir= target_dir),
            LogValidationSamples(
                base_dir=cfg.save_dir,
                num_samples=cfg.num_val_exemples,
                norm = full_data.dataset.norm,
                min_ir=full_data.dataset.min_val_ir,
                max_ir=full_data.dataset.max_val_ir,
                min_sar=full_data.dataset.min_val_sar,
                max_sar=full_data.dataset.max_val_sar,
                mean_sar = full_data.dataset.mean_val_sar,
                std_sar = full_data.dataset.std_val_sar,
                mean_ir = full_data.dataset.mean_val_ir,
                std_ir = full_data.dataset.std_val_ir,
                cmap_ir=cmap_ir,
                cmap_sar=cmap_sar,
                start_epoch=cfg.start_epoch,    
                mask= dictio["mask_sar"]
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
    model = create_model(image_size=cfg.img_size, dropout=cfg.dropout).to(fabric.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)   #add regularisation

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    # Prepare for Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # --- Training Loop ---
    train_loss_history = []
    val_loss_history = []
    for epoch in range(cfg.num_epochs):
        logger.info(f"===== Epoch {epoch+1}/{cfg.num_epochs} =====")
        
        train_loss, train_metrics = train_one_epoch(fabric, model, train_loader, optimizer, metrics)
        val_loss, val_metrics = validate(fabric, model, val_loader, metrics)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        scheduler.step()

        fabric.call(
            "on_validation_epoch_end",
            val_loss=val_loss,
            model=model,
            fabric=fabric
        )

        if epoch ==0 or ((epoch + 1) >= cfg.start_epoch and (epoch + 1) % cfg.plot_interval == 0):
                fabric.call(
                "on_validation_plots",
                model=model,
                epoch=epoch,
                dataloader=val_loader,
                device=fabric.device
            )
                print("----- plots saved")

        fabric.print(
            f"📊 Epoch {epoch+1}: Train Loss={train_loss:.6f}, "
            f"Val Loss={val_loss:.6f}, "
            f"LR={scheduler.get_last_lr()[0]:.6f}, "
            # f"Train metrics={train_metrics}, Val metrics={val_metrics}"
        )
    history_df = pd.DataFrame({
    "train_loss": train_loss_history,
    "val_loss": val_loss_history
    })
    

    # Construire le chemin du CSV
    csv_path = target_dir / "training_history.csv"
    history_df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plot_path = os.path.join(target_dir, "loss_history.png")
    plt.savefig(plot_path)
    plt.close()

    # save the config 
    config_path = "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/config.yaml"
    shutil.copy(config_path,os.path.join(target_dir,"config_used.yaml"))
    dataprep.compute_global_distributions(model, val_loader, fabric.device,full_data.dataset.mean_val_sar, full_data.dataset.std_val_sar,
                                          save_dir=os.path.join(target_dir, "distributions"))
    logger.info("🎯 Training Complete!")


if __name__ == "__main__":
    from src.IR_to_SAR.ML_IR_SAR.config import IR_SAR_Config
    cfg = IR_SAR_Config.from_yaml("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/config.yaml")
    main(cfg,test=False)