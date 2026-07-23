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
import pickle as pkl

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

def train_one_epoch(fabric, model, dataloader, optimizer, cfg, scheduler=None):
    model.train()

    total_loss = 0.0
    total_l_pix = 0.0
    total_l_grad = 0.0
    total_l_radial = 0.0

    BIN_EDGES = compute_bin_edges_quantiles(
        dataloader,
        device=fabric.device,
        num_bins=5
    )

    BIN_WEIGHTS, BIN_PROBS, BIN_COUNTS = compute_bin_weights_from_loader(
        train_loader=dataloader,
        bin_edges=BIN_EDGES,
        device=fabric.device,
        alpha=0.5
    )

    for x, sar, mask, infos in tqdm(dataloader, desc="Training"):
        x = x.to(fabric.device)
        sar = sar.to(fabric.device)
        mask = mask.to(fabric.device)

        if cfg.channel_dropout:
            x = model_ir_sar.apply_random_channel_dropout(
                x,
                drop_prob=cfg.channel_drop_prob,
                min_keep_channels=cfg.min_keep_channels,
                protect_channels=getattr(cfg, "protect_channels", None),
            )

        if cfg.conditional_model:
            shear_infos = torch.stack([
                torch.as_tensor(d["shear"], dtype=torch.float32)
                for d in infos
            ]).to(fabric.device)

        optimizer.zero_grad()

        if not cfg.conditional_model:
            pred = model(x, timestep=0).sample
        else:
            pred = model.forward(x, timestep=0, cond=shear_infos).sample

        # Exclure tous les pixels SAR non finis
        finite_sar = torch.isfinite(sar)
        mask = mask * finite_sar.to(mask.dtype)
        # Remplacer NaN et Inf avant toute opération arithmétique
        sar_valid = torch.nan_to_num(
            sar,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        # Mettre réellement à zéro les zones exclues
        sar_valid = sar_valid * mask
        pred_valid = pred

        loss, l_pix, l_grad, l_radial = combined_sar_loss(
            sar_valid,
            pred_valid,
            mask,
            w_pix=cfg.w_pix,
            w_grad=cfg.w_grad,
            w_radial=cfg.w_radial,
            bin_edges=BIN_EDGES,
            bin_weights=BIN_WEIGHTS,
            use_weighted_pix=True
        )

        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.detach().item())
        total_l_pix += float(l_pix)
        total_l_grad += float(l_grad)
        total_l_radial += float(l_radial)

    n_batches = len(dataloader)

    train_loss = torch.tensor(total_loss / n_batches, device=fabric.device)
    avg_l_pix = torch.tensor(total_l_pix / n_batches, device=fabric.device)
    avg_l_grad = torch.tensor(total_l_grad / n_batches, device=fabric.device)
    avg_l_radial = torch.tensor(total_l_radial / n_batches, device=fabric.device)

    train_loss = fabric.all_reduce(train_loss, reduce_op="mean")
    avg_l_pix = fabric.all_reduce(avg_l_pix, reduce_op="mean")
    avg_l_grad = fabric.all_reduce(avg_l_grad, reduce_op="mean")
    avg_l_radial = fabric.all_reduce(avg_l_radial, reduce_op="mean")

    return (
        train_loss.item(),
        avg_l_pix.item(),
        avg_l_grad.item(),
        avg_l_radial.item()
    )

def validate(fabric, model, dataloader, cfg):
    model.eval()

    total_loss = 0.0
    total_l_pix = 0.0
    total_l_grad = 0.0
    total_l_radial = 0.0

    BIN_EDGES = compute_bin_edges_quantiles(
        dataloader,
        device=fabric.device,
        num_bins=5
    )

    BIN_WEIGHTS, BIN_PROBS, BIN_COUNTS = compute_bin_weights_from_loader(
        train_loader=dataloader,
        bin_edges=BIN_EDGES,
        device=fabric.device,
        alpha=0.5
    )

    with torch.no_grad():
        for x, sar, mask, infos in tqdm(dataloader, desc="Validating"):
            x = x.to(fabric.device)
            sar = sar.to(fabric.device)
            mask = mask.to(fabric.device)

            if cfg.conditional_model:
                shear_infos = torch.stack([
                    torch.as_tensor(d["shear"], dtype=torch.float32)
                    for d in infos
                ]).to(fabric.device)

            if not cfg.conditional_model:
                pred = model(x, timestep=0).sample
            else:
                pred = model.forward(x, timestep=0, cond=shear_infos).sample

            # Exclure tous les pixels SAR non finis
            finite_sar = torch.isfinite(sar)
            mask = mask * finite_sar.to(mask.dtype)
            # Remplacer NaN et Inf avant toute opération arithmétique
            sar_valid = torch.nan_to_num(
                sar,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            # Mettre réellement à zéro les zones exclues
            sar_valid = sar_valid * mask
            pred_valid = pred

            loss, l_pix, l_grad, l_radial = combined_sar_loss(
                sar_valid,
                pred_valid,
                mask,
                w_pix=cfg.w_pix,
                w_grad=cfg.w_grad,
                w_radial=cfg.w_radial,
                bin_edges=BIN_EDGES,
                bin_weights=BIN_WEIGHTS,
                use_weighted_pix=True
            )

            total_loss += float(loss.detach().item())
            total_l_pix += float(l_pix)
            total_l_grad += float(l_grad)
            total_l_radial += float(l_radial)

    n_batches = len(dataloader)

    val_loss = torch.tensor(total_loss / n_batches, device=fabric.device)
    avg_l_pix = torch.tensor(total_l_pix / n_batches, device=fabric.device)
    avg_l_grad = torch.tensor(total_l_grad / n_batches, device=fabric.device)
    avg_l_radial = torch.tensor(total_l_radial / n_batches, device=fabric.device)

    val_loss = fabric.all_reduce(val_loss, reduce_op="mean")
    avg_l_pix = fabric.all_reduce(avg_l_pix, reduce_op="mean")
    avg_l_grad = fabric.all_reduce(avg_l_grad, reduce_op="mean")
    avg_l_radial = fabric.all_reduce(avg_l_radial, reduce_op="mean")

    return (
        val_loss.item(),
        avg_l_pix.item(),
        avg_l_grad.item(),
        avg_l_radial.item()
    )


def main(cfg: config.IR_SAR_Config, test=False):
    cfg.use_residu = False
    cfg.use_flow_matching = False
    cfg.protect_channels = cfg.protect_channels if not cfg.add_era5 else [cfg.in_channels - 1]
    cfg.early_stop_patience = 2 if cfg.code_test else cfg.early_stop_patience
    cfg.batch_size = cfg.batch_size if not cfg.code_test else 1
    cfg.out_channels = 12 if cfg.temporal_mode else cfg.out_channels

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
        target_dir
    )

def train(fabric, cfg, full_data, target_dir):
    
    cfg.in_channels = full_data.dataset.X_train.shape[1]
    cfg.anggrek_test = False if cfg.temporal_mode else cfg.anggrek_test

    if fabric.is_global_zero:
        logger.info(f"Running on device: {fabric.device}")
        logger.info(f"num_epochs = {cfg.num_epochs}")
        logger.info(f"code_test = {cfg.code_test}")
        logger.info(f"batch_size = {cfg.batch_size}")
        logger.info(f"early_stop_patience = {cfg.early_stop_patience}")

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

    in_channels = (
        train_ds.X.shape[1]
        if isinstance(train_ds.X, np.ndarray)
        else train_ds.X[0].shape[0]
    )

    model = model_ir_sar.create_model(
        cfg=cfg,
        conditional_model=cfg.conditional_model,
        in_channels=in_channels
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.fm_lr if cfg.use_flow_matching else cfg.learning_rate,
        weight_decay=1e-3
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epochs,
        eta_min=1e-6
    )

    model, optimizer = fabric.setup(model, optimizer)

    test_loader_full_for_plot = DataLoader(
                                            test_ds,
                                            batch_size=cfg.batch_size,
                                            shuffle=False,
                                            collate_fn=dataprep.custom_collate
                                            )
    if cfg.code_test : 
        train_loader_full_for_plot = DataLoader(
                                                train_ds,
                                                batch_size=cfg.batch_size,
                                                shuffle=False,
                                                collate_fn=dataprep.custom_collate
                                                )

    if cfg.anggrek_test:
        train_loader, val_loader, test_loader, anggrek_loader = fabric.setup_dataloaders(
            train_loader,
            val_loader,
            test_loader,
            anggrek_loader
        )
        anggrek_loader_full_for_plot = DataLoader(
                                                anggrek_ds,
                                                batch_size=cfg.batch_size,
                                                shuffle=False,
                                                collate_fn=dataprep.custom_collate
                                                )
    else:
        train_loader, val_loader, test_loader = fabric.setup_dataloaders(
            train_loader,
            val_loader,
            test_loader
        )

    train_loss_history = []
    val_loss_history = []
    pix2pix_loss_history = []
    gradient_loss_history = []
    radial_loss_history = []

    best_val_loss = float("inf")
    patience_counter = 0
    stop_training = False

    best_ckpt_path = target_dir / "best_regression_model.pt"

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

        train_loss, l_pix, l_grad, l_radial = train_one_epoch(
            fabric,
            model,
            train_loader,
            optimizer,
            scheduler=scheduler,
            cfg=cfg
        )

        val_loss, l_pix_val, l_grad_val, l_radial_val = validate(
            fabric,
            model,
            val_loader,
            cfg=cfg
        )

        scheduler.step()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        pix2pix_loss_history.append((l_pix, l_pix_val))
        gradient_loss_history.append((l_grad, l_grad_val))
        radial_loss_history.append((l_radial, l_radial_val))

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

                fabric.print(f"✅ Best regression model saved: val_loss={val_loss:.6f}")

            else:
                patience_counter += 1
                fabric.print(
                    f"Early stopping counter: "
                    f"{patience_counter}/{cfg.early_stop_patience}"
                )

            if epoch == 0 and not cfg.temporal_mode:
                fabric.print("📸 Plot at epoch 1")

                plot_callback.on_validation_plots(
                    model=model,
                    epoch=epoch,
                    dataloader=(
                        [test_loader_full_for_plot if not cfg.code_test else train_loader_full_for_plot, anggrek_loader_full_for_plot]
                        if cfg.anggrek_test
                        else [test_loader_full_for_plot if not cfg.code_test else train_loader_full_for_plot]
                    ),
                    device=fabric.device
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
                
                with open(os.path.join(target_dir, "model.pkl"),"wb") as f:
                    pkl.dump(model, f)

                if not cfg.temporal_mode:
                    fabric.print("📸 Final plot using BEST model")

                    plot_callback.on_validation_plots(
                        model=model,
                        epoch=epoch,
                        dataloader=(
                            [test_loader_full_for_plot if not cfg.code_test else train_loader_full_for_plot, anggrek_loader_full_for_plot]
                            if cfg.anggrek_test
                            else [test_loader_full_for_plot if not cfg.code_test else train_loader_full_for_plot]
                        ),
                        device=fabric.device
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