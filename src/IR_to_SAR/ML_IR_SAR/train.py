# This script wil be used to test a simple U-NET for a regression simple (IR -> SAR)

import os
import gc
import sys
os.chdir("/scale/user/mtannaou/alternance")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

print("Python Path:", sys.path)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
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
    def __init__(self,test=False,size=256, norm = "z_score", barycenter = "no" ,augmentation = False, drop_nan_100 = True,input_channels=None,
                 data_path=None):
        self.norm = norm
        
        dataset = prep_dataset.PrepareDataSet(size=size, norm= norm, barycenter= barycenter, drop_nan_100=drop_nan_100,input_channels=input_channels,
                                              pkl_file=data_path)
        self.X = dataset.X
        self.sar = dataset.sar   
        self.dataset = dataset     
        print("Data preparation finished")
        print(self.X.shape, np.expand_dims(self.sar, axis=1).shape)

    def __len__(self):    #number of observations
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]          # already (C,H,W)
        sar = self.sar[idx]      # (H,W)

        # Ensure target has channel dimension
        sar = torch.tensor(sar, dtype=torch.float32).unsqueeze(0)

        return torch.tensor(X, dtype=torch.float32), sar


class PairedDataset(Dataset):
    def __init__(self, X, sar_array, mask, cyclone_id, sar_time):
        self.X = X
        self.sar = sar_array
        self.mask = mask
        self.cyclone_id =  cyclone_id
        self.sar_time = sar_time
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.sar[idx], dtype=torch.float32),
            torch.tensor(self.mask[idx], dtype=torch.float32),
            self.cyclone_id[idx],
            self.sar_time[idx]
        )





# =============================
# 🧠 2) Train / Validate functions
# =============================
def train_one_epoch(fabric, model, dataloader, optimizer, metrics):
    model.train()
    total_loss = 0
    metrics.reset()

    for x, sar, mask, _, _ in tqdm(dataloader, desc="Training"):
        x, sar, mask= x.to(fabric.device), sar.to(fabric.device), mask.to(fabric.device)
        optimizer.zero_grad()

        # Forward pass
        pred = model(x, timestep=0).sample  # (B,1,H,W)

        # Mask des pixels valides
        # mask = torch.isfinite(sar)

        sar_valid = sar.nan_to_num()
        pred_valid = pred

        # Losses
        if sar_valid.ndim == 3:
            sar_valid = sar_valid.unsqueeze(1)
        loss_mse = F.l1_loss(pred_valid*mask, sar_valid*mask)
        loss_ssim = 1 - ssim(pred_valid, sar_valid)

        loss = 1.0 * loss_mse + 0.0 * loss_ssim

        # Backpropagation
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        metrics.update(pred_valid*mask, sar_valid*mask)

    return total_loss / len(dataloader), metrics.compute()


def validate(fabric, model, dataloader, metrics):
    model.eval()
    total_loss = 0
    metrics.reset()

    with torch.no_grad():
        for x, sar, mask, _, _ in tqdm(dataloader, desc="Validating"):
            x, sar, mask = x.to(fabric.device), sar.to(fabric.device), mask.to(fabric.device)

            pred = model(x, timestep=0).sample

            # mask = torch.isfinite(sar)
            sar_valid = sar.nan_to_num()
            pred_valid = pred
            if sar_valid.ndim == 3:
                sar_valid = sar_valid.unsqueeze(1)

            loss_mse = F.l1_loss(pred_valid*mask, sar_valid*mask)
            loss_ssim = 1 - ssim(pred_valid, sar_valid)

            loss = 1.0 * loss_mse + 0.0 * loss_ssim

            total_loss += loss.item()
            metrics.update(pred_valid*mask, sar_valid*mask)

    return total_loss / len(dataloader), metrics.compute()


# =============================
# 🚀 3) Main Training Lightning Fabric
# =============================
def custom_collate(batch):
    """
    batch = [
        (x, sar, mask, cyclone_id, sar_time),
        ...
    ]
    """
    xs        = torch.stack([item[0] for item in batch])
    sars      = torch.stack([item[1] for item in batch])
    masks     = torch.stack([item[2] for item in batch])

    cyclone_ids = [item[3] for item in batch]   # stay as list of strings
    sar_times   = [item[4] for item in batch]   # stay as list of strings

    return xs, sars, masks, cyclone_ids, sar_times

def main(cfg: IR_SAR_Config,test=False):
    logger.info(f"Starting training with config:\n{cfg.__dict__}")
    stop_training = False

    # --- Dataset full ---
    
    full_data = IRSARDataset(test=test, size=cfg.img_size, norm=cfg.norm, barycenter=cfg.barycenter, drop_nan_100=cfg.drop_nan_sar,
                             input_channels=cfg.input_channels,
                             data_path = cfg.data_path)
    X_all, sar_all = full_data.dataset.X, full_data.dataset.sar

    if cfg.barycenter == "yes" : 
        dx = full_data.dataset.dx_sar
        dy = full_data.dataset.dy_sar
    
    # --- Split ---
    dictio = dataprep.train_val_test_split_random(
        np.array(X_all), np.array(sar_all),
        train_size=cfg.train_split,
        val_size=cfg.val_split,
        test_size=cfg.test_split,
        augmentation=cfg.augmentation,
        mask_sar = full_data.dataset.mask_sar,
        cyclone_ids=full_data.dataset.cyclone_ids,
        sar_times= full_data.dataset.sar_time
    )
    
    train_ds = PairedDataset(*dictio["train"],dictio["mask_sar_train"], dictio["cyclone_id_train"], dictio["sar_time_train"])  #X (multi-channel input), SAR target
    val_ds   = PairedDataset(*dictio["val"],dictio["mask_sar_val"], dictio["cyclone_id_val"], dictio["sar_time_val"])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate)

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
                norm = cfg.norm,
                mean_X=full_data.dataset.mean_X,
                mean_sar=full_data.dataset.mean_sar,
                std_X=full_data.dataset.std_X,
                std_sar=full_data.dataset.std_sar,
                cmap_ir=cmap_ir,
                cmap_sar=cmap_sar,
                start_epoch=cfg.start_epoch,    
                mask_train = dictio["mask_sar_train"],
                mask_val = full_data.dataset.mask_sar[dictio["val_index"]],
                cyclone_ids= full_data.dataset.cyclone_ids,
                sar_times= full_data.dataset.sar_time
            )
        ],
    )
    fabric.launch()
    dataprep.visualize_dataset_statistics(dictio,target_dir,full_data.dataset.mask_sar)

    # --- Metrics ---
    metrics = torchmetrics.MetricCollection({
        "mse": torchmetrics.MeanSquaredError(),
        "mae": torchmetrics.MeanAbsoluteError(),
    }).to(fabric.device)

    # --- Model & Optimizer & Scheduler ---
    in_channels = train_ds.X.shape[1] if isinstance(train_ds.X, np.ndarray) else train_ds.X[0].shape[0]
    model = create_model(
        image_size=cfg.img_size,
        dropout=cfg.dropout,
        in_channels=in_channels,       #
        out_channels=cfg.out_channels
    ).to(fabric.device)
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

        # torch.cuda.empty_cache()
        # gc.collect()
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        scheduler.step()

        fabric.call(
            "on_validation_epoch_end",
            val_loss=val_loss,
            epoch=epoch,
            model=model,
            fabric=fabric
        )

        if epoch == 0:
            fabric.print("📸 Plot at epoch 1")

            fabric.call(
                "on_validation_plots",
                model=model,
                epoch=epoch,
                dataloader=[train_loader, val_loader],
                device=fabric.device
            )
            print("----- plots saved")



        fabric.print(
            f"📊 Epoch {epoch+1}: Train Loss={train_loss:.6f}, "
            f"Val Loss={val_loss:.6f}, "
            f"LR={scheduler.get_last_lr()[0]:.6f}, "
            # f"Train metrics={train_metrics}, Val metrics={val_metrics}"
        )
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
      
        for callback in fabric._callbacks:
            if getattr(callback, "should_stop", False):
                fabric.print("\n⛔ Early stopping activated — stopping training.")
                early_stop_epoch = epoch
                stop_training = True
                break

        if stop_training or epoch == cfg.num_epochs - 1:
            best_ckpt_path = target_dir / "best_regression_model.pt"

            if best_ckpt_path.exists():
                fabric.print("✅ Loading best model for final visualization...")
                ckpt = torch.load(best_ckpt_path, map_location=fabric.device)
                model.load_state_dict(ckpt["model"])
            else:
                fabric.print("⚠️ Best checkpoint not found, using last model.")

            fabric.print("📸 Final plot using BEST model")

            fabric.call(
                "on_validation_plots",
                model=model,
                epoch=epoch,   # dernier epoch
                dataloader=[train_loader, val_loader],
                device=fabric.device
            )
            break


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

    
    
    logger.info("🎯 Training Complete!")


if __name__ == "__main__":
    from src.IR_to_SAR.ML_IR_SAR.config import IR_SAR_Config
    cfg = IR_SAR_Config.from_yaml("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/config.yaml")
    main(cfg,test=False)