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


from datetime import datetime, timedelta
from loguru import logger
import lightning as L  # used for the callbacks
from lightning.fabric.strategies import DDPStrategy   

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from importlib import reload
import torch
torch.set_float32_matmul_precision('high')
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

import torch.nn.functional as F
import torchmetrics
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path


from src.IR_to_SAR.ML_IR_SAR.losses import *
from src.visualisation.utils_colormap import CMAP
cmap_ir , cmap_sar = CMAP.cira_ir(), CMAP.cmap_sar()
import src.IR_to_SAR.ML_IR_SAR.model as model_ir_sar
reload(model_ir_sar)
import src.IR_to_SAR.ML_IR_SAR.callbacks as callbacks
reload(callbacks)
import src.IR_to_SAR.ML_IR_SAR.config as config
reload(config)
import src.IR_to_SAR.data_preparation.prepare_dataset as prep_dataset
reload(prep_dataset)
import src.IR_to_SAR.data_preparation.data_preprocessing as dataprep
reload(dataprep)
import src.IR_to_SAR.data_preparation.distribution_data_visualisation as distdata
reload(distdata)

from src.set_seed import set_seed
set_seed(0)


class IRSARDataset(Dataset):
    """
    Dataset for IR → SAR prediction.
    IR shape:  (N, H_ir, W_ir)
    SAR shape: (N, H_ir, W_ir)
    """
    def __init__(self,
                 target_dir = None,
                 cfg=None):
    
        self.norm = cfg.norm
        dataset = prep_dataset.PrepareDataSet(
                                              target_dir = target_dir,
                                              cfg=cfg)
        self.dataset = dataset     

    def __len__(self):    #number of observations
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]          # already (C,H,W)
        sar = self.sar[idx]      # (H,W)
        sar = torch.tensor(sar, dtype=torch.float32).unsqueeze(0) if sar.ndim == 2 else torch.tensor(sar, dtype=torch.float32)
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


def train_one_epoch(
    fabric,
    fm_model,
    regression_model,
    dataloader,
    optimizer,
    residual_mean: float,
    residual_std: float,
    scheduler=None,
    scheduler_name=None,
    cfg=None
):
    fm_model.train()
    regression_model.eval()
    total_loss = 0.0
    resid_mean_t = torch.tensor(residual_mean, dtype=torch.float32)
    resid_std_t  = torch.tensor(residual_std,  dtype=torch.float32)
    for x, sar, mask, infos in tqdm(dataloader, desc="Resid-FM train", leave=True):
        x    = x.to(fabric.device)
        sar  = sar.to(fabric.device)
        mask = mask.to(fabric.device)
        if cfg.channel_dropout:
                        x = model_ir_sar.apply_random_channel_dropout(
                        x,
                        drop_prob=cfg.channel_drop_prob,
                        min_keep_channels=cfg.min_keep_channels,
                        protect_channels=getattr(cfg, "protect_channels", None),
                    )
        if sar.ndim == 3:
            sar  = sar.unsqueeze(1)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        with torch.no_grad():
            t0        = torch.zeros(x.shape[0], device=fabric.device)
            mean_pred = regression_model(x, t0).sample   # (B, 1, H, W)
        rm = resid_mean_t.to(fabric.device)
        rs = resid_std_t.to(fabric.device)

        residual = (sar - mean_pred).nan_to_num(0.0)
        x_1      = (residual - rm) / rs     # normalised residual ∈ ~N(0, 1)
        B   = x_1.shape[0]
        z   = torch.randn_like(x_1)
        t   = torch.rand(B, device=fabric.device)
        x_t = t.view(-1,1,1,1) * x_1 + (1 - t.view(-1,1,1,1)) * z
        model_input   = torch.cat([x_t, mean_pred], dim=1)   # (B, 2, H, W)

        optimizer.zero_grad()
        pred_velocity = fm_model(model_input, t).sample
        true_velocity = x_1 - z
        valid = (mask > 0) & sar.isfinite()
        valid = valid.expand_as(pred_velocity)
        if valid.sum() == 0:
            continue
        loss  = F.mse_loss(pred_velocity[valid], true_velocity[valid])
        fabric.backward(loss)
        fabric.clip_gradients(fm_model, optimizer, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)

def validate(
    fabric,
    fm_model,
    regression_model,
    dataloader,
    residual_mean: float,
    residual_std: float,
):
    fm_model.eval()
    regression_model.eval()
    total_loss = 0.0
    n_batches = 0
    resid_mean_t = torch.tensor(residual_mean, dtype=torch.float32, device=fabric.device)
    resid_std_t  = torch.tensor(residual_std, dtype=torch.float32, device=fabric.device)
    with torch.no_grad():
        for x, sar, mask, infos in tqdm(dataloader, desc="Resid-FM val", leave=True):
            x    = x.to(fabric.device)
            sar  = sar.to(fabric.device)
            mask = mask.to(fabric.device)
            if sar.ndim == 3:
                sar = sar.unsqueeze(1)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            t0 = torch.zeros(x.shape[0], device=fabric.device)
            mean_pred = regression_model(x, t0).sample   # (B, 1, H, W)
            residual = (sar - mean_pred).nan_to_num(0.0)
            x_1 = (residual - resid_mean_t) / resid_std_t
            B = x_1.shape[0]
            z = torch.randn_like(x_1)
            t = torch.rand(B, device=fabric.device)
            x_t = t.view(-1, 1, 1, 1) * x_1 + (1 - t.view(-1, 1, 1, 1)) * z
            model_input = torch.cat([x_t, mean_pred], dim=1)   # (B, 2, H, W)
            pred_velocity = fm_model(model_input, t).sample
            true_velocity = x_1 - z
            valid = (mask > 0) & sar.isfinite()
            valid = valid.expand_as(pred_velocity)
            if valid.sum() == 0:
                continue
            loss = F.mse_loss(pred_velocity[valid], true_velocity[valid])
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main(cfg: config.IR_SAR_Config, test=False):
    cfg.use_residu = True
    cfg.use_flow_matching = True
    cfg.protect_channels = cfg.protect_channels if not cfg.add_era5 else [cfg.in_channels - 1]
    cfg.early_stop_patience = 5 if cfg.code_test else cfg.early_stop_patience
    cfg.batch_size = cfg.batch_size if not cfg.code_test else 8

    logger.info(f"Starting training with config:\n{cfg.__dict__}")

    base_dir = Path(cfg.save_dir)
    i = 1
    while (base_dir / f"train_ir_sar_{i}").exists():
        i += 1

    target_dir = base_dir / f"train_ir_sar_{i}"
    os.makedirs(target_dir, exist_ok=True)

    full_data = IRSARDataset(
        target_dir=target_dir,
        cfg=cfg
    )

    train_ds = PairedDataset(
        *(full_data.dataset.X_train, full_data.dataset.sar_train),
        full_data.dataset.mask_train,
        full_data.dataset.infos_train
    )

    val_ds = PairedDataset(
        *(full_data.dataset.X_val, full_data.dataset.sar_val),
        full_data.dataset.mask_val,
        full_data.dataset.infos_val
    )

    test_ds = PairedDataset(
        *(full_data.dataset.X_test, full_data.dataset.sar_test),
        full_data.dataset.mask_test,
        full_data.dataset.infos_test
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=dataprep.custom_collate
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=dataprep.custom_collate
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=dataprep.custom_collate
    )

    if isinstance(cfg.devices, int):
        n_devices = cfg.devices
    else:
        n_devices = len(cfg.devices)

    fabric = L.Fabric(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=DDPStrategy(
            start_method="spawn",
            process_group_backend="gloo",
            timeout=timedelta(minutes=120)
        ) if n_devices > 1 else "auto",
    )

    fabric.launch(
        train,
        cfg,
        full_data,
        target_dir,
        train_loader,
        val_loader,
        test_loader
    )

def train(fabric, cfg, full_data, target_dir, train_loader, val_loader, test_loader):
    
    cfg.in_channels = full_data.dataset.X_train.shape[1]

    if fabric.is_global_zero:
        logger.info(f"Running on device: {fabric.device}")
        logger.info(f"num_epochs = {cfg.num_epochs}")
        logger.info(f"code_test = {cfg.code_test}")
        logger.info(f"batch_size = {cfg.batch_size}")
        logger.info(f"early_stop_patience = {cfg.early_stop_patience}")

    anggrek_loader = None

    if cfg.anggrek_test:
        anggrek_ds = PairedDataset(
            *(full_data.dataset.X_anggrek, full_data.dataset.X_anggrek),
            full_data.dataset.X_anggrek,
            full_data.dataset.infos_anggrek
        )

        anggrek_loader = DataLoader(
            anggrek_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=dataprep.custom_collate
        )

    model = model_ir_sar.create_fm_residual_model(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.fm_lr,
        weight_decay=1e-3
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epochs,
        eta_min=1e-6
    )

    model, optimizer = fabric.setup(model, optimizer)

    if cfg.anggrek_test:
        train_loader, val_loader, test_loader, anggrek_loader = fabric.setup_dataloaders(
            train_loader,
            val_loader,
            test_loader,
            anggrek_loader
        )
    else:
        train_loader, val_loader, test_loader = fabric.setup_dataloaders(
            train_loader,
            val_loader,
            test_loader
        )

    best_reg_model = model_ir_sar.load_regression_model(
        cfg.best_regression_model_pt,
        cfg=cfg
    )
    best_reg_model = best_reg_model.to(fabric.device)
    best_reg_model.eval()

    json_file_name = "residual_stats_test" if cfg.code_test else "residual_stats"
    path_resid_stats = target_dir / f"{json_file_name}.json"

    if fabric.is_global_zero:
        if path_resid_stats.exists():
            resid_stats = dataprep.load_residual_stats(path_resid_stats)
        else:
            stats = dataprep.compute_residual_stats(
                train_loader=train_loader,
                regression_model=best_reg_model,
                device=fabric.device
            )

            dataprep.save_residual_stats(
                stats,
                path_resid_stats
            )

            resid_stats = stats

    fabric.barrier()

    resid_stats = dataprep.load_residual_stats(path_resid_stats)

    train_loss_history = []
    val_loss_history = []
    pix2pix_loss_history = []
    gradient_loss_history = []
    radial_loss_history = []

    best_val_loss = float("inf")
    patience_counter = 0
    stop_training = False

    best_ckpt_path = target_dir / "best_fm_model.pt"

    plot_callback = callbacks.LogValidationSamples(
        base_dir=cfg.save_dir,
        mean_X=full_data.dataset.mean_X,
        mean_sar=full_data.dataset.mean_sar,
        std_X=full_data.dataset.std_X,
        std_sar=full_data.dataset.std_sar,
        cmap_ir=cmap_ir,
        cmap_sar=cmap_sar,
        mask_train=full_data.dataset.mask_train,
        mask_val=full_data.dataset.mask_val,
        mask_test=full_data.dataset.mask_test,
        infos_train=full_data.dataset.infos_train,
        infos_val=full_data.dataset.infos_val,
        infos_test=full_data.dataset.infos_test,
        target_dir=target_dir,
        cfg=cfg
    )

    for epoch in range(cfg.num_epochs):

        if fabric.is_global_zero:
            logger.info(f"===== Epoch {epoch + 1}/{cfg.num_epochs} =====")

        train_loss = train_one_epoch(
            fabric=fabric,
            fm_model=model,
            regression_model=best_reg_model,
            dataloader=train_loader,
            optimizer=optimizer,
            residual_mean=resid_stats["mean"],
            residual_std=resid_stats["std"],
            scheduler=scheduler,
            scheduler_name=cfg.scheduler,
            cfg=cfg
        )

        val_loss = validate(
            fabric,
            model,
            best_reg_model,
            val_loader,
            resid_stats["mean"],
            resid_stats["std"]
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if fabric.is_global_zero:
            fabric.print(
                f"📊 Epoch {epoch + 1}: "
                f"Train Loss={train_loss:.6f}, "
                f"Val Loss={val_loss:.6f}, "
                f"LR={scheduler.get_last_lr()[0]:.6f}"
            )

            if val_loss < best_val_loss - cfg.early_stop_delta:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save(
                    {
                        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "cfg": cfg.__dict__,
                    },
                    best_ckpt_path
                )

                fabric.print(f"✅ Best model saved: val_loss={val_loss:.6f}")

            else:
                patience_counter += 1
                fabric.print(
                    f"Early stopping counter: "
                    f"{patience_counter}/{cfg.early_stop_patience}"
                )

            if epoch == 0:
                fabric.print("📸 Plot at epoch 1")

                plot_callback.on_validation_plots(
                    model=model,
                    epoch=epoch,
                    dataloader=(
                        [test_loader if not cfg.code_test else train_loader, anggrek_loader]
                        if cfg.anggrek_test
                        else [test_loader if not cfg.code_test else train_loader]
                    ),
                    device=fabric.device,
                    reg_model=best_reg_model,
                    resid_stats=resid_stats
                )

                fabric.print("----- plots saved")

            if patience_counter >= cfg.early_stop_patience:
                fabric.print("\n⛔ Early stopping activated — stopping training.")
                stop_training = True

        fabric.barrier()

        stop_tensor = torch.tensor(
            int(stop_training),
            device=fabric.device
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(stop_tensor, src=0)

        stop_training = bool(stop_tensor.item())

        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

        if stop_training or epoch == cfg.num_epochs - 1:
            if fabric.is_global_zero:
                if best_ckpt_path.exists():
                    fabric.print("✅ Loading best model for final visualization...")
                    ckpt = torch.load(best_ckpt_path, map_location=fabric.device)

                    state_dict = ckpt["model"]
                    if hasattr(model, "module"):
                        model.module.load_state_dict(state_dict)
                    else:
                        model.load_state_dict(state_dict)

                else:
                    fabric.print("⚠️ Best checkpoint not found, using last model.")

                fabric.print("📸 Final plot using BEST model")

                plot_callback.on_validation_plots(
                    model=model,
                    epoch=epoch,
                    dataloader=(
                        [test_loader if not cfg.code_test else train_loader, anggrek_loader]
                        if cfg.anggrek_test
                        else [test_loader if not cfg.code_test else train_loader]
                    ),
                    device=fabric.device,
                    reg_model=best_reg_model,
                    resid_stats=resid_stats
                )

                distdata.training_completed(
                    cfg,
                    train_loss_history,
                    val_loss_history,
                    pix2pix_loss_history,
                    gradient_loss_history,
                    radial_loss_history,
                    target_dir
                )

            fabric.barrier()
            break

if __name__ == "__main__":
    
    config_path = "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/config.yaml"
    cfg = config.IR_SAR_Config.from_yaml(config_path)
    main(cfg, test=False)
