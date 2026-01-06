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

from src.IR_to_SAR.ML_IR_SAR.losses import combined_sar_loss,compute_bin_weights_from_loader,compute_bin_edges_quantiles

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
                 data_path=None,train_split=None,val_split=None,test_split=None,target_dir = None):
        self.norm = norm
        
        dataset = prep_dataset.PrepareDataSet(size=size, norm= norm, barycenter= barycenter, drop_nan_100=drop_nan_100,input_channels=input_channels,
                                              pkl_file=data_path,train_split=train_split,val_split=val_split,test_split=test_split,
                                              augmentation=augmentation,target_dir = target_dir)
         
        self.dataset = dataset     
        print("Data preparation finished")
        # print(self.X.shape, np.expand_dims(self.sar, axis=1).shape)

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
def train_one_epoch(fabric, model, dataloader, optimizer, metrics,w_pix=0.1,w_grad=0.0,w_radial=0.0):
    model.train()
    total_loss = 0
    metrics.reset()

    BIN_EDGES = compute_bin_edges_quantiles(dataloader, device=fabric.device, num_bins=5)
    BIN_WEIGHTS, BIN_PROBS, BIN_COUNTS = compute_bin_weights_from_loader(
        train_loader=dataloader, bin_edges=BIN_EDGES, device=fabric.device, alpha=0.5
    )

    print("BIN_EDGES:", BIN_EDGES)
    print("BIN_PROBS:", BIN_PROBS)
    print("BIN_WEIGHTS:", BIN_WEIGHTS)


    for x, sar, mask, _ in tqdm(dataloader, desc="Training"):
        x, sar, mask= x.to(fabric.device), sar.to(fabric.device), mask.to(fabric.device)
        optimizer.zero_grad()

        # Forward pass
        pred = model(x, timestep=0).sample  # (B,1,H,W)
        sar_valid = sar.nan_to_num()
        pred_valid = pred

        # compute weights
        

        loss, l_pix, l_grad, l_radial = combined_sar_loss(
                                                            sar_valid, pred_valid, mask,
                                                            w_pix=w_pix, w_grad=w_grad, w_radial=w_radial,
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

    return total_loss / len(dataloader), metrics.compute(), l_pix.cpu(), l_grad.cpu(), l_radial.cpu()


def validate(fabric, model, dataloader, metrics,w_pix=0.1,w_grad=0.0,w_radial=0.0):
    model.eval()
    total_loss = 0
    metrics.reset()

    with torch.no_grad():
        for x, sar, mask, _ in tqdm(dataloader, desc="Validating"):
            x, sar, mask = x.to(fabric.device), sar.to(fabric.device), mask.to(fabric.device)

            pred = model(x, timestep=0).sample

            # mask = torch.isfinite(sar)
            sar_valid = sar.nan_to_num()
            pred_valid = pred

            loss,l_pix,l_grad,l_radial = combined_sar_loss(sar_valid,pred_valid,mask,
                                     w_pix=w_pix,
                                     w_grad=w_grad,
                                     w_radial=w_radial)
            
            if sar_valid.ndim == 3:
                sar_valid = sar_valid.unsqueeze(1)

            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            total_loss += loss.item()
            metrics.update(pred_valid*mask, sar_valid*mask)

    return total_loss / len(dataloader), metrics.compute(), l_pix.cpu(), l_grad.cpu(),l_radial.cpu()


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
                             train_split=cfg.train_split,val_split=cfg.val_split,test_split=cfg.test_split,target_dir = target_dir,augmentation=cfg.augmentation)
    # X_all, sar_all = full_data.dataset.X, full_data.dataset.sar

    if cfg.barycenter == "yes" : 
        dx = full_data.dataset.dx_sar
        dy = full_data.dataset.dy_sar

    
    
    
    
    train_ds = PairedDataset(*(full_data.dataset.X_train,full_data.dataset.sar_train),full_data.dataset.dictio["mask_sar_train"], full_data.dataset.dictio["infos_train"])  #X (multi-channel input), SAR target
    val_ds   = PairedDataset(*(full_data.dataset.X_val,full_data.dataset.sar_val),full_data.dataset.dictio["mask_sar_val"], full_data.dataset.dictio["infos_val"])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate)

    # --- Fabric init with callbacks ---   #QUentin
    
    
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
                mask_train = full_data.dataset.dictio["mask_sar_train"],
                mask_val = full_data.dataset.mask_sar[full_data.dataset.dictio["val_index"]],
                infos = full_data.dataset.infos,
                target_dir = target_dir,
                radial_mean = full_data.dataset.radial_profil
            )
        ],
    )
    fabric.launch()
    dataprep.visualize_dataset_statistics(full_data.dataset.dictio,target_dir,full_data.dataset.mask_sar)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-3)   #add regularisation

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs//2)

    # Prepare for Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # --- Training Loop ---
    train_loss_history = []
    val_loss_history = []
    pix2pix_loss_history = []
    gradient_loss_history = []
    radial_loss_history = []
    for epoch in range(cfg.num_epochs):
        logger.info(f"===== Epoch {epoch+1}/{cfg.num_epochs} =====")
        
        train_loss, train_metrics,l_pix, l_grad, l_radial = train_one_epoch(fabric, model, train_loader, optimizer, metrics,
                                                    w_pix=cfg.w_pix,
                                                    w_grad=cfg.w_grad,
                                                    w_radial=cfg.w_radial)
        val_loss, val_metrics, l_pix_val,l_grad_val, l_radial_val = validate(fabric, model, val_loader, metrics,
                                         w_pix=cfg.w_pix,
                                         w_grad=cfg.w_grad,
                                         w_radial=cfg.w_radial)

        # torch.cuda.empty_cache()
        # gc.collect()
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        pix2pix_loss_history.append((l_pix,l_pix_val))
        gradient_loss_history.append((l_grad,l_grad_val))
        radial_loss_history.append((l_radial,l_radial_val))


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
            
            f"LR={scheduler.get_last_lr()[0]:.6f}, "
            f"pix2pix loss={l_pix.item():.6f},  "
            f"gradient loss={l_grad.item():.6f},  "
            f"radial loss={l_radial.item():.6f}  "
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
    "val_loss": val_loss_history,
    "pix2pix_history": pix2pix_loss_history,
    "gradient_loss_history" : gradient_loss_history,
    "radial_loss_history" : radial_loss_history

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
    #3 losses 
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


if __name__ == "__main__":
    from src.IR_to_SAR.ML_IR_SAR.config import IR_SAR_Config
    cfg = IR_SAR_Config.from_yaml("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/config.yaml")
    main(cfg,test=False)