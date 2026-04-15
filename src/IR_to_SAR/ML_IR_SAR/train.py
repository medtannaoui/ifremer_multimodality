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

from src.IR_to_SAR.ML_IR_SAR.losses import *

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
from src.IR_to_SAR.ML_IR_SAR.model import create_model,create_fm_model_direct,load_regression_model,create_fm_residual_model
from src.IR_to_SAR.ML_IR_SAR.callbacks import EarlyStopping,ModelCheckpoint,LogValidationSamples

import src.IR_to_SAR.ML_IR_SAR.config as config
reload(config)
from src.IR_to_SAR.ML_IR_SAR.config import IR_SAR_Config
import src.IR_to_SAR.data_preparation.prepare_dataset as prep_dataset
reload(prep_dataset)

from src.IR_to_SAR.data_preparation.data_preprocessing import compute_residual_stats,load_residual_stats,save_residual_stats




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
                 data_path=None,train_split=None,val_split=None,test_split=None,target_dir = None, input_data="norm", output_data="sar",
                 conditional_model=None,anggrek_test = False,log_wind=None,irwin_channels=1,regrid_ir=False,
                 ir_smoothing=False,add_era5=False,cfg=None):
        self.norm = norm
        
        dataset = prep_dataset.PrepareDataSet(size=size, norm= norm, barycenter= barycenter, drop_nan_100=drop_nan_100,input_channels=input_channels,
                                              pkl_file=data_path,train_split=train_split,val_split=val_split,test_split=test_split,
                                              augmentation=augmentation,target_dir = target_dir, input_data=input_data, output_data=output_data,
                                              conditional_model=conditional_model,anggrek_test=anggrek_test,log_wind=log_wind,irwin_channels=irwin_channels,
                                              regrid_ir=regrid_ir,ir_smoothing=ir_smoothing,add_era5=add_era5,cfg=cfg)
         
        self.dataset = dataset     
        print("Data preparation finished")


    def __len__(self):    #number of observations
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]          # already (C,H,W)
        sar = self.sar[idx]      # (H,W)

        # Ensure target has channel dimension
        sar = torch.tensor(sar, dtype=torch.float32).unsqueeze(0)

        return torch.tensor(X, dtype=torch.float32), sar


class PairedDataset(Dataset):
    def __init__(self, X, sar_array, mask, infos):
        self.X = X
        self.sar = sar_array
        self.mask = mask
        self.infos = infos
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
        torch.tensor(self.X[idx], dtype=torch.float32),
        torch.tensor(self.sar[idx], dtype=torch.float32),
        torch.tensor(self.mask[idx], dtype=torch.float32),
        self.infos[idx]  
    )





# =============================
# 🧠 2) Train / Validate functions
# =============================
def train_one_epoch(fabric, model, dataloader, optimizer, metrics, cfg, scheduler=None):
    model.train()
    total_loss = 0
    metrics.reset()

    BIN_EDGES = compute_bin_edges_quantiles(dataloader, device=fabric.device, num_bins=5)
    BIN_WEIGHTS, BIN_PROBS, BIN_COUNTS = compute_bin_weights_from_loader(
        train_loader=dataloader, bin_edges=BIN_EDGES, device=fabric.device, alpha=0.5
    )

    print("BIN_WEIGHTS:", BIN_WEIGHTS)

    for x, sar, mask, infos in tqdm(dataloader, desc="Training"):  #infos is a dictionanry
        x, sar, mask= x.to(fabric.device), sar.to(fabric.device), mask.to(fabric.device)
        if cfg.conditional_model:
            
            shear_infos = torch.stack([
            torch.as_tensor(d["shear"], dtype=torch.float32)
                                for d in infos
                            ]).to(fabric.device)

        optimizer.zero_grad()

        
        # Forward pass
        if not cfg.conditional_model:
            pred = model(x, timestep=0).sample  # (B,1,H,W)
        else:
            pred = model.forward(x, timestep=0, cond=shear_infos).sample

        B,H,W = sar.shape
        hs = slice(H//2 - H//4, H//2 + H//4)
        ws = slice(W//2 - W//4, W//2 + W//4)

        sar_valid  = sar.nan_to_num()[:, hs, ws] if cfg.crop_sar else sar.nan_to_num()
        pred_valid = pred[:, :, hs, ws] if cfg.crop_sar else pred
        if cfg.crop_sar:
            mask = mask[:, :, hs, ws] if mask.ndim == 4 else mask[:, hs, ws] 
        # compute weights

        loss, l_pix, l_grad, l_radial = combined_sar_loss(
                                                            sar_valid, pred_valid, mask,
                                                            w_pix=cfg.w_pix, w_grad=cfg.w_grad, w_radial=cfg.w_radial,
                                                            bin_edges=BIN_EDGES, bin_weights=BIN_WEIGHTS,
                                                            use_weighted_pix=True
                                                        )

        if sar_valid.ndim == 3:
            sar_valid = sar_valid.unsqueeze(1)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        # Backpropagation
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=1.0)
        optimizer.step()
        

        total_loss += loss.item()
        metrics.update(pred_valid*mask, sar_valid*mask)

    return total_loss / len(dataloader), metrics.compute(), l_pix, l_grad, l_radial


def validate(fabric, model, dataloader, metrics, cfg):
    model.eval()
    total_loss = 0
    metrics.reset()
    BIN_EDGES = compute_bin_edges_quantiles(dataloader, device=fabric.device, num_bins=5)
    BIN_WEIGHTS, BIN_PROBS, BIN_COUNTS = compute_bin_weights_from_loader(
        train_loader=dataloader, bin_edges=BIN_EDGES, device=fabric.device, alpha=0.5
    )

    with torch.no_grad():
        for x, sar, mask, infos in tqdm(dataloader, desc="Validating"):
            x, sar, mask = x.to(fabric.device), sar.to(fabric.device), mask.to(fabric.device)

            if not cfg.conditional_model:
                pred = model(x, timestep=0).sample  # (B,1,H,W)
            else:
                shear_infos = torch.stack([
                torch.as_tensor(d["shear"], dtype=torch.float32)
                                for d in infos
                            ]).to(fabric.device)
                pred = model.forward(x, timestep=0, cond=shear_infos).sample

            B,H,W = sar.shape
            hs = slice(H//2 - H//4, H//2 + H//4)
            ws = slice(W//2 - W//4, W//2 + W//4)

            sar_valid  = sar.nan_to_num()[:, hs, ws] if cfg.crop_sar else sar.nan_to_num()
            pred_valid = pred[:, :, hs, ws] if cfg.crop_sar else pred
            if cfg.crop_sar : 
                mask = mask[:, :, hs, ws] if mask.ndim == 4 else mask[:, hs, ws]

            loss,l_pix,l_grad,l_radial = combined_sar_loss(sar_valid,pred_valid,mask,
                                     w_pix=cfg.w_pix,
                                     w_grad= cfg.w_grad,
                                     w_radial=cfg.w_radial,
                                     bin_edges=BIN_EDGES, bin_weights=BIN_WEIGHTS,
              
                                                            use_weighted_pix=True)
            

            if sar_valid.ndim == 3:
                sar_valid = sar_valid.unsqueeze(1)

            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            total_loss += loss.item()
            metrics.update(pred_valid*mask, sar_valid*mask)

    return total_loss / len(dataloader), metrics.compute(), l_pix, l_grad,l_radial

def train_fm_epoch_direct(fabric, model, dataloader, optimizer, scheduler=None,
                          scheduler_name=None):
    """
    One flow matching training epoch — direct approach.

    Args:
        fabric:         Lightning Fabric instance (handles device + amp)
        model:          FM UNet (in_channels = C_IR + 1)
        dataloader:     returns (x, sar, mask, infos) batches
        optimizer:      PyTorch optimizer
        scheduler:      optional LR scheduler
        scheduler_name: "onecycle" | "cosin" | "expo" | "reduceplateau" | None

    Returns:
        mean loss over the epoch (float)
    """
    model.train()
    total_loss = 0.0

    for x, sar, mask, infos in tqdm(dataloader, desc="FM train", leave=False):
        x    = x.to(fabric.device)
        sar  = sar.to(fabric.device)
        mask = mask.to(fabric.device)

        B = x.shape[0]

        # Ensure sar and mask are (B, 1, H, W)
        if sar.ndim == 3:
            sar  = sar.unsqueeze(1)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # ── Flow matching forward ──────────────────────────────────────
        x_1 = sar.nan_to_num(0.0)                       # NaN → 0 (excluded by mask)
        z   = torch.randn_like(x_1)                     # source noise
        t   = torch.rand(B, device=fabric.device)       # t ~ U[0, 1]

        # Linear interpolation (the FM "forward process")
        x_t = t.view(-1, 1, 1, 1) * x_1 + (1 - t.view(-1, 1, 1, 1)) * z

        # Concatenate noisy SAR with IR conditioning
        model_input = torch.cat([x_t, x], dim=1)        # (B, C_IR+1, H, W)

        optimizer.zero_grad()

        pred_velocity = model(model_input, t).sample     # (B, 1, H, W)
        true_velocity = x_1 - z

        # Loss on valid SAR pixels only
        valid = (mask > 0).expand_as(pred_velocity)
        loss  = F.mse_loss(pred_velocity[valid], true_velocity[valid])

        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=1.0)
        optimizer.step()

        # OneCycle scheduler steps per batch
        if scheduler is not None and scheduler_name == "onecycle":
            scheduler.step()

        total_loss += loss.item()

    # Epoch-level scheduler step for other types
    if scheduler is not None and scheduler_name not in (None, "onecycle"):
        scheduler.step()

    return total_loss / max(len(dataloader), 1)


def validate_fm_direct(fabric, model, dataloader, stats, num_inference_steps=50):
    """
    Flow matching validation.

    Returns a dict with:
      - 'loss': mean FM velocity loss (primary metric for early stopping / checkpoint)
      - 'mae_mean': pixel-space MAE of the ensemble mean vs SAR target
    """
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    n_batches  = 0

    with torch.no_grad():
        for x, sar, mask, infos in tqdm(dataloader, desc="FM val", leave=False):
            x    = x.to(fabric.device)
            sar  = sar.to(fabric.device)
            mask = mask.to(fabric.device)

            B = x.shape[0]

            if sar.ndim == 3:
                sar  = sar.unsqueeze(1)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            # ── FM velocity loss (same as training) ───────────────────
            x_1 = sar.nan_to_num(0.0)
            z   = torch.randn_like(x_1)
            t   = torch.rand(B, device=fabric.device)
            x_t = t.view(-1,1,1,1)*x_1 + (1-t.view(-1,1,1,1))*z

            model_input   = torch.cat([x_t, x], dim=1)
            pred_velocity = model(model_input, t).sample
            true_velocity = x_1 - z

            valid = (mask > 0).expand_as(pred_velocity)
            loss  = F.mse_loss(pred_velocity[valid], true_velocity[valid])
            total_loss += loss.item()

            # # ── Pixel-space MAE: one sample per validation example ────
            # z_infer = torch.randn_like(x_1)
            # x_pred  = _euler_ode(model, z_infer, x, num_inference_steps, fabric.device)

            # # Denormalise if stats provided
            # sar_mean = torch.tensor(stats["mean"], device=fabric.device).view(1,1,1,1)
            # sar_std  = torch.tensor(stats["std"],  device=fabric.device).view(1,1,1,1)
            # sar_phys = sar  * sar_std + sar_mean
            # pred_phys = x_pred * sar_std + sar_mean

            # mae = (pred_phys - sar_phys).abs()[valid].mean()
            # total_mae += mae.item()
            n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n
   
def train_fm_residual_epoch(
    fabric,
    fm_model,
    regression_model,
    dataloader,
    optimizer,
    residual_mean: float,
    residual_std: float,
    scheduler=None,
    scheduler_name=None,
):
    """
    One training epoch for residual FM.

    Args:
        fabric:           Lightning Fabric
        fm_model:         residual FM UNet (in_channels=2)
        regression_model: frozen regression UNet
        dataloader:       returns (x, sar, mask, infos)
        optimizer:        for fm_model only (regression is frozen)
        residual_mean:    pre-computed mean of (SAR - regression) on training set
        residual_std:     pre-computed std  of (SAR - regression) on training set
    """
    fm_model.train()
    regression_model.eval()
    total_loss = 0.0

    resid_mean_t = torch.tensor(residual_mean, dtype=torch.float32)
    resid_std_t  = torch.tensor(residual_std,  dtype=torch.float32)

    for x, sar, mask, infos in tqdm(dataloader, desc="Resid-FM train", leave=False):
        x    = x.to(fabric.device)
        sar  = sar.to(fabric.device)
        mask = mask.to(fabric.device)

        if sar.ndim == 3:
            sar  = sar.unsqueeze(1)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # ── Step 1: frozen regression (mean field) ─────────────────────
        with torch.no_grad():
            t0        = torch.zeros(x.shape[0], device=fabric.device)
            mean_pred = regression_model(x, t0).sample   # (B, 1, H, W)

        # ── Step 2: normalised residual ────────────────────────────────
        rm = resid_mean_t.to(fabric.device)
        rs = resid_std_t.to(fabric.device)

        residual = (sar - mean_pred).nan_to_num(0.0)
        x_1      = (residual - rm) / rs     # normalised residual ∈ ~N(0, 1)

        # ── Step 3: FM interpolation in residual space ─────────────────
        B   = x_1.shape[0]
        z   = torch.randn_like(x_1)
        t   = torch.rand(B, device=fabric.device)
        x_t = t.view(-1,1,1,1) * x_1 + (1 - t.view(-1,1,1,1)) * z

        # Model input: [noisy residual, mean prediction]
        model_input   = torch.cat([x_t, mean_pred], dim=1)   # (B, 2, H, W)

        optimizer.zero_grad()

        pred_velocity = fm_model(model_input, t).sample
        true_velocity = x_1 - z

        # Mask: valid SAR pixels only
        valid = (mask > 0) & sar.isfinite()
        valid = valid.expand_as(pred_velocity)
        loss  = F.mse_loss(pred_velocity[valid], true_velocity[valid])

        fabric.backward(loss)
        fabric.clip_gradients(fm_model, optimizer, max_norm=1.0)
        optimizer.step()

        if scheduler is not None and scheduler_name == "onecycle":
            scheduler.step()

        total_loss += loss.item()

    if scheduler is not None and scheduler_name not in (None, "onecycle"):
        scheduler.step()

    return total_loss / max(len(dataloader), 1)

def validate_fm_residual_epoch(
    fabric,
    fm_model,
    regression_model,
    dataloader,
    residual_mean: float,
    residual_std: float,
):
    """
    One validation epoch for residual FM.

    Args:
        fabric:           Lightning Fabric
        fm_model:         residual FM UNet (in_channels=2)
        regression_model: frozen regression UNet
        dataloader:       returns (x, sar, mask, infos)
        residual_mean:    mean of (SAR - regression prediction) on training set
        residual_std:     std  of (SAR - regression prediction) on training set

    Returns:
        mean validation loss over the epoch (float)
    """
    fm_model.eval()
    regression_model.eval()
    total_loss = 0.0
    n_batches = 0

    resid_mean_t = torch.tensor(residual_mean, dtype=torch.float32, device=fabric.device)
    resid_std_t  = torch.tensor(residual_std, dtype=torch.float32, device=fabric.device)

    with torch.no_grad():
        for x, sar, mask, infos in tqdm(dataloader, desc="Resid-FM val", leave=False):
            x    = x.to(fabric.device)
            sar  = sar.to(fabric.device)
            mask = mask.to(fabric.device)

            if sar.ndim == 3:
                sar = sar.unsqueeze(1)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            # ── Step 1: frozen regression prediction ───────────────────
            t0 = torch.zeros(x.shape[0], device=fabric.device)
            mean_pred = regression_model(x, t0).sample   # (B, 1, H, W)

            # ── Step 2: normalised residual target ─────────────────────
            residual = (sar - mean_pred).nan_to_num(0.0)
            x_1 = (residual - resid_mean_t) / resid_std_t

            # ── Step 3: FM interpolation in residual space ─────────────
            B = x_1.shape[0]
            z = torch.randn_like(x_1)
            t = torch.rand(B, device=fabric.device)
            x_t = t.view(-1, 1, 1, 1) * x_1 + (1 - t.view(-1, 1, 1, 1)) * z

            # Model input: [noisy residual, mean prediction]
            model_input = torch.cat([x_t, mean_pred], dim=1)   # (B, 2, H, W)

            pred_velocity = fm_model(model_input, t).sample
            true_velocity = x_1 - z

            # Valid pixels only
            valid = (mask > 0) & sar.isfinite()
            valid = valid.expand_as(pred_velocity)

            loss = F.mse_loss(pred_velocity[valid], true_velocity[valid])

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)

# =============================
# 🚀 3) Main Training Lightning Fabric
# =============================
def custom_collate(batch):
    """
    batch = [
        (x, sar, mask, infos_dict),
        ...
    ]
    """
    xs    = torch.stack([item[0] for item in batch])
    sars  = torch.stack([item[1] for item in batch])
    masks = torch.stack([item[2] for item in batch])

    # infos reste une liste de dictionnaires
    infos = [item[3] for item in batch]

    return xs, sars, masks, infos


def main(cfg: IR_SAR_Config,test=False):
    cfg.use_residu = cfg.use_residu if cfg.use_flow_matching else False
    import matplotlib.pyplot as plt
    logger.info(f"Starting training with config:\n{cfg.__dict__}")
    stop_training = False

    base_dir = Path(cfg.save_dir)

    i = 1
    while (base_dir / f"train_ir_sar_{i}").exists():
        i += 1

    target_dir = base_dir / f"train_ir_sar_{i}"

    os.makedirs(target_dir, exist_ok=True)

    # --- Dataset full ---
    
    full_data = IRSARDataset(test=test, size=cfg.img_size, norm=cfg.norm, barycenter=cfg.barycenter, drop_nan_100=cfg.drop_nan_sar,
                             input_channels=cfg.input_channels,
                             data_path = cfg.data_path,
                             train_split=cfg.train_split,val_split=cfg.val_split,test_split=cfg.test_split,
                             target_dir = target_dir,augmentation=cfg.augmentation,input_data=cfg.input_data,output_data=cfg.output_data,
                             conditional_model = cfg.conditional_model,anggrek_test=cfg.anggrek_test,log_wind=cfg.log_wind,irwin_channels=cfg.irwin_channels,
                             regrid_ir=cfg.regrid_ir,ir_smoothing=cfg.ir_smoothing,add_era5=cfg.add_era5,cfg=cfg)
    # X_all, sar_all = full_data.dataset.X, full_data.dataset.sar

    
    train_ds = PairedDataset(*(full_data.dataset.X_train,full_data.dataset.sar_train),full_data.dataset.mask_train, full_data.dataset.infos_train)  #X (multi-channel input), SAR target
    val_ds   = PairedDataset(*(full_data.dataset.X_val,full_data.dataset.sar_val),full_data.dataset.mask_val, full_data.dataset.infos_val)
    test_ds   = PairedDataset(*(full_data.dataset.X_test,full_data.dataset.sar_test),full_data.dataset.mask_test, full_data.dataset.infos_test)
    
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=custom_collate)
    if cfg.anggrek_test:
        anggrek_ds   = PairedDataset(*(full_data.dataset.X_anggrek,full_data.dataset.X_anggrek),full_data.dataset.X_anggrek, full_data.dataset.infos_anggrek)
        anggrek_loader = DataLoader(anggrek_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate)

    # --- Fabric init with callbacks ---   #QUentin
    
    ckpt_filename = (
                        "best_fm_resid_model.pt" if cfg.use_residu
                        else "best_fm_model.pt" if cfg.use_flow_matching
                        else "best_regression_model.pt"
                    )
    fabric = L.Fabric(
        accelerator=cfg.accelerator,
        devices= cfg.devices,
        strategy= "auto",
        callbacks=[
            EarlyStopping(patience=cfg.early_stop_patience, min_delta=cfg.early_stop_delta),
            ModelCheckpoint(cfg.save_dir, filename=ckpt_filename, target_dir= target_dir),
            LogValidationSamples(
                base_dir= cfg.save_dir,
                mean_X=full_data.dataset.mean_X,
                mean_sar=full_data.dataset.mean_sar,
                std_X=full_data.dataset.std_X,
                std_sar=full_data.dataset.std_sar,
                cmap_ir=cmap_ir,
                cmap_sar=cmap_sar,   
                mask_train = full_data.dataset.mask_train,
                mask_val = full_data.dataset.mask_val,
                mask_test=full_data.dataset.mask_test,
                infos_train = full_data.dataset.infos_train,
                infos_val = full_data.dataset.infos_val,
                infos_test = full_data.dataset.infos_test,
                target_dir = target_dir,
                cfg=cfg
            )
        ],
    )
    fabric.launch()
    # dataprep.visualize_dataset_statistics(full_data.dataset.dictio,target_dir,full_data.dataset.mask_sar)

    # --- Metrics ---
    metrics = torchmetrics.MetricCollection({
        "mse": torchmetrics.MeanSquaredError(),
        "mae": torchmetrics.MeanAbsoluteError(),
    }).to(fabric.device)

    # --- Model & Optimizer & Scheduler ---
    in_channels = train_ds.X.shape[1] if isinstance(train_ds.X, np.ndarray) else train_ds.X[0].shape[0]
    if cfg.use_flow_matching:
        if not cfg.use_residu:
            model = create_fm_model_direct(cfg, in_channels_ir=in_channels)
        else :
            model = create_fm_residual_model(cfg)
        
    else : 
        model = create_model(
            cfg=cfg,
            conditional_model=cfg.conditional_model,
            in_channels=in_channels
        ).to(fabric.device)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.fm_lr if cfg.use_flow_matching else cfg.learning_rate, weight_decay=1e-3)   #add regularisation

    if cfg.scheduler == "cosin":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
        


    elif cfg.scheduler == "expo":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.98
    )
        
    elif cfg.scheduler == "reduceplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",         
            factor=0.5,          
            patience=5,          
            threshold=1e-4,
            min_lr=1e-6,
            verbose=True
        )
        
    elif cfg.scheduler == "onecycle" : 
        steps_per_epoch = len(train_loader)  # ton DataLoader train
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.fm_lr if cfg.use_flow_matching else cfg.learning_rate,     # peak lr
            epochs=cfg.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,                # 10% warmup
            div_factor=10.0,              # lr initial = max_lr/div_factor
            final_div_factor=1e4          # lr final = max_lr/final_div_factor
        )


    # Prepare for Fabric
    model, optimizer = fabric.setup(model, optimizer)
    if cfg.anggrek_test:
        train_loader, val_loader, test_loader, anggrek_loader = fabric.setup_dataloaders(
                                                            train_loader, val_loader, test_loader, anggrek_loader
                                                            )
    else : 
        train_loader, val_loader, test_loader = fabric.setup_dataloaders(
                                                            train_loader, val_loader, test_loader
                                                            )


    # --- Training Loop ---
    train_loss_history = []
    val_loss_history = []
    pix2pix_loss_history = []
    gradient_loss_history = []
    radial_loss_history = []
    best_reg_model = None
    if cfg.use_residu and cfg.use_flow_matching:
        best_reg_model = load_regression_model(cfg.best_regression_model_pt,cfg=cfg)
        best_reg_model = best_reg_model.to(fabric.device)
        
        path_resid_stats = "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/flow_matching_residual/residual_stats.json" if not cfg.code_test else "residual_stats.json"
        if os.path.exists(path_resid_stats):
            resid_stats = load_residual_stats(path_resid_stats)
        else : 
            stats = compute_residual_stats(train_loader=train_loader, regression_model=best_reg_model,device=fabric.device)
            save_residual_stats(stats,path_resid_stats)
            resid_stats  = stats

    for epoch in range(cfg.num_epochs):
        logger.info(f"===== Epoch {epoch+1}/{cfg.num_epochs} =====")
        if cfg.use_flow_matching:
            if not cfg.use_residu:                                     
                train_loss = train_fm_epoch_direct(                       
                    fabric, model, train_loader, optimizer,               
                    scheduler, cfg.scheduler,                              
                )                                                         
                val_loss = validate_fm_direct(
                                    fabric, model, val_loader, 0, cfg.fm_num_inference_steps
                                ) 
            else : 
                train_loss = train_fm_residual_epoch(fabric=fabric,fm_model=model,regression_model=best_reg_model,
                                                     dataloader=train_loader,
                                                     optimizer=optimizer,residual_mean=resid_stats["mean"],
                                                     residual_std=resid_stats["std"],
                                                     scheduler=scheduler,scheduler_name=cfg.scheduler)
                val_loss = validate_fm_residual_epoch(fabric,model,best_reg_model,val_loader,resid_stats["mean"],resid_stats["std"])
        else:
            train_loss, train_metrics,l_pix, l_grad, l_radial = train_one_epoch(fabric, model, train_loader, optimizer, metrics,
                                                        scheduler=scheduler,
                                                        cfg = cfg)
            
            val_loss, val_metrics, l_pix_val,l_grad_val, l_radial_val = validate(fabric, model, val_loader, metrics,
                                                        cfg = cfg)

        # torch.cuda.empty_cache()
        # gc.collect()
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        if not cfg.use_flow_matching:
            pix2pix_loss_history.append((l_pix,l_pix_val))
            gradient_loss_history.append((l_grad,l_grad_val))
            radial_loss_history.append((l_radial,l_radial_val))


        if cfg.scheduler == "reduceplateau":
                scheduler.step(val_loss)   # needs metric
        elif cfg.scheduler in ["cosin", "expo"]:
            scheduler.step()           # epoch-based, no metric
        elif cfg.scheduler == "onecycle":
            pass  # stepped per-batch inside train_one_epoch

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
                dataloader= [train_loader, val_loader, test_loader] if not cfg.anggrek_test else   [train_loader, val_loader, test_loader, anggrek_loader],
                device=fabric.device,
                reg_model = best_reg_model if cfg.use_residu else None,
                resid_stats = resid_stats if cfg.use_residu else None
            )
            print("----- plots saved")



        fabric.print(
            f"📊 Epoch {epoch+1}: Train Loss={train_loss:.6f}, "
            
            f"LR={scheduler.get_last_lr()[0]:.6f}, "
            f"Val Loss={val_loss:.6f},  "
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
            ckpt_name = (
                            "best_fm_resid_model.pt" if cfg.use_residu
                            else "best_fm_model.pt" if cfg.use_flow_matching
                            else "best_regression_model.pt"
                        )
            best_ckpt_path = target_dir / ckpt_name

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
                dataloader=[train_loader, val_loader, test_loader, anggrek_loader] if cfg.anggrek_test else [train_loader, val_loader, test_loader],
                device=fabric.device,
                reg_model = best_reg_model if cfg.use_residu else None,
                resid_stats = resid_stats if cfg.use_residu else None

            )
            break
        


    

    plt.figure()
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plot_path = os.path.join(target_dir, "loss_history.png" if not cfg.use_flow_matching else "fm_loss_history.png")
    plt.savefig(plot_path)
    plt.close()
    #3 losses 
    if not cfg.use_flow_matching:
        history_df = pd.DataFrame({
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "pix2pix_history": pix2pix_loss_history,
        "gradient_loss_history" : gradient_loss_history,
        "radial_loss_history" : radial_loss_history

        })
        

        # Construire le chemin du CSV
        csv_path = target_dir / "training_history.csv" 
        
        history_df.to_csv(csv_path, index=False)
        pix_train = [x[0] for x in pix2pix_loss_history]
        pix_val   = [x[1] for x in pix2pix_loss_history]

        grad_train = [x[0] for x in gradient_loss_history]
        grad_val   = [x[1] for x in gradient_loss_history]

        rad_train = [x[0] for x in radial_loss_history]
        rad_val   = [x[1] for x in radial_loss_history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

        # --- Pix2Pix ---
        axes[0].plot(pix_train, label="Train")
        axes[0].plot(pix_val, label="Val")
        axes[0].set_title("Pix2Pix Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # --- Gradient ---
        axes[1].plot(grad_train, label="Train")
        axes[1].plot(grad_val, label="Val")
        axes[1].set_title("Gradient Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True)

        # --- Radial ---
        axes[2].plot(rad_train, label="Train")
        axes[2].plot(rad_val, label="Val")
        axes[2].set_title("Radial Loss (Vmax)")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True)

        plt.suptitle("Loss Components Evolution", fontsize=14)
        plt.tight_layout()
        plot_path = os.path.join(target_dir, "3_losses_history.png")
        plt.savefig(plot_path)
        plt.close()

    logger.info("🎯 Training Complete!")
    dst_cfg = os.path.join(target_dir, "config.yaml")
    config_path ="/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/config.yaml"

    if hasattr(cfg, "config_path") and config_path and os.path.exists(config_path):
        shutil.copy(config_path, dst_cfg)
        logger.info(f"📄 Config copied to: {dst_cfg}")
    else:
        # fallback: save cfg content as text if the yaml path isn't available
        with open(os.path.join(target_dir, "config_fallback.txt"), "w") as f:
            for k, v in cfg.__dict__.items():
                f.write(f"{k}: {v}\n")
        logger.warning("⚠️ config.yaml path not found, saved config_fallback.txt instead")


if __name__ == "__main__":
    from src.IR_to_SAR.ML_IR_SAR.config import IR_SAR_Config

    config_path = "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/config.yaml"
    cfg = IR_SAR_Config.from_yaml(config_path)

    main(cfg, test=False)