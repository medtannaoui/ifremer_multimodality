import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    """
    Callback to stop training when a monitored metric has stopped improving.
    """

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.epochs_without_improvement = 0
        self.best_val_loss = float("inf")
        self.should_stop = False

    def on_validation_epoch_end(self, val_loss, **kwargs):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            print("Early stopping triggered.")




class ModelCheckpoint:
    """
    Callback to save the model after an epoch if the validation loss improved.
    """

    def __init__(self, output_dir, filename="best_regression_model.pt"):
        self.output_path = Path(output_dir) / filename
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(self, val_loss, model, fabric, **kwargs):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            # Use fabric.save to handle distributed saving correctly
            fabric.save(self.output_path, {"model": model.state_dict()})
            fabric.print(f"Validation loss improved. Saved model to {self.output_path}")





class LogValidationSamples:
    """
    Callback to log IR → SAR predictions after epoch 20,
    saving per-sample plots in a unique train directory.
    """

    def __init__(self, base_dir, min_ir, max_ir, min_sar, max_sar,mean_sar,std_sar,mean_ir,std_ir,norm,
             num_samples=4, start_epoch=20, every_n_epochs=1,
             cmap_ir="gray", cmap_sar="viridis",mask = None):

        self.base_dir = Path(base_dir)
        self.output_dir = self._create_unique_dir(self.base_dir)

        self.num_samples = num_samples
        self.start_epoch = start_epoch
        self.every_n_epochs = every_n_epochs  
        self.cmap_ir = cmap_ir
        self.cmap_sar = cmap_sar

        self.min_ir = min_ir
        self.max_ir = max_ir
        self.min_sar = min_sar
        self.max_sar = max_sar
        self.mean_ir = mean_ir
        self.mean_sar = mean_sar
        self.std_sar = std_sar
        self.std_ir = std_ir
        self.norm = norm
        self.mask = mask


    def _create_unique_dir(self, base_dir):
        i = 1
        while (base_dir / f"train_ir_sar_{i}").exists():
            i += 1
        new_dir = base_dir / f"train_ir_sar_{i}"
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir

    def denormalize(self, tensor, min_val, max_val,mean_val, std_val, eps = 1e-10):
        """Convert back from [0,1] to real physical values."""
        if self.norm == "z_score":
            return ((tensor * (std_val + eps)) - mean_val)
        else : 
            return tensor * (max_val - min_val + eps) + min_val

    def log_batch(self, model, batch, epoch, device):
        """
        Saves IR / Predicted SAR / Real SAR visualizations 
        for a single epoch AFTER self.start_epoch.
        """
        if epoch < self.start_epoch or (epoch - self.start_epoch) % self.every_n_epochs != 0:
            return  # 🛑 Do nothing before epoch threshold

        model.eval()
        ir, sar = batch
        ir = ir.to(device)
        sar = sar.to(device)

        with torch.no_grad():
            pred = model(ir, timestep=0).sample

        # ---- Dé-normalisation ----
        ir_denorm   = self.denormalize(ir, self.min_ir, self.max_ir, self.mean_ir, self.std_ir)
        sar_denorm  = self.denormalize(sar, self.min_sar, self.max_sar, self.mean_sar, self.std_sar)
        pred_denorm = self.denormalize(pred, self.min_sar, self.max_sar,self.mean_sar, self.std_sar)

        # ---- Convert to numpy ----
        ir_np   = ir_denorm.squeeze(1).cpu().numpy()
        sar_np  = sar_denorm.squeeze(1).cpu().numpy()
        pred_np = pred_denorm.squeeze(1).cpu().numpy()

        num = min(self.num_samples, ir_np.shape[0])

        # ---- Save EACH sample as separate PNG ----
        for i in range(num):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # IR Input
            axes[0].imshow(ir_np[i], cmap=self.cmap_ir)
            axes[0].set_title(f"IRWIN")
            axes[0].axis("off")

            
            # Real SAR
            sar_vis = np.where(self.mask[i] == 1, sar_np[i], np.nan)
            axes[1].imshow(sar_vis, cmap=self.cmap_sar)
            axes[1].set_title(f"Target SAR (knots)")
            axes[1].axis("off")

            # Predicted SAR
            pred_vis = np.where(self.mask[i] == 1, pred_np[i], np.nan)
            axes[2].imshow(pred_vis, cmap=self.cmap_sar)
            axes[2].set_title(f"Predicted SAR (knots)")
            axes[2].axis("off")

            fig.suptitle(f"Sample {i} — Epoch {epoch + 1}", fontsize=14)
            plt.tight_layout()

            # 📸 Save per-sample image
            os.makedirs(os.path.join(Path(self.output_dir ), "samples"), exist_ok=True)
            save_path =  os.path.join(os.path.join(Path(self.output_dir ), "samples"), f"sample_{i}_epoch_{epoch + 1}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

            print(f"💾 Saved: {save_path}")
    
    def on_validation_plots(self, model, epoch, dataloader, device):
        """
        Called manually using fabric.call(...)
        """
        print(f"📸 Logging validation samples at epoch {epoch}")
        batch = next(iter(dataloader))  # get one batch
        self.log_batch(model, batch, epoch, device)

