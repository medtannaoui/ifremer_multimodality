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

    def __init__(self, output_dir, filename="best_regression_model.pt", target_dir= None):
        self.output_path = Path(output_dir) / target_dir / filename
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

    def __init__(self, base_dir, mean_X, std_X, mean_sar, std_sar, norm,
             num_samples=4, start_epoch=20, every_n_epochs=1,
             cmap_ir="gray", cmap_sar="viridis",mask = None, num_epochs=0, cyclone_ids=None, sar_times=None, distance_km=None):

        self.base_dir = Path(base_dir)
        self.output_dir = self._create_unique_dir(self.base_dir)

        self.num_samples = num_samples
        self.start_epoch = start_epoch
        self.every_n_epochs = every_n_epochs  
        self.cmap_ir = cmap_ir
        self.cmap_sar = cmap_sar

        self.mean_sar = mean_sar
        self.mean_X = mean_X
        self.std_sar = std_sar
        self.std_X = std_X
        
        self.norm = norm
        self.mask = mask
        self.num_epochs = num_epochs
        self.cyclone_ids = cyclone_ids
        self.sar_times = sar_times
        


    def _create_unique_dir(self, base_dir):
        i = 1
        while (base_dir / f"train_ir_sar_{i}").exists():
            i += 1
        new_dir = base_dir / f"train_ir_sar_{i}"
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir

    def denormalize(self, tensor,mean_val, std_val, eps = 1e-10):
        """Convert back from [0,1] to real physical values."""
        
        return ((tensor * (std_val + eps)) + mean_val)
        

    def log_batch(self, model, batch, epoch, device):
        # Skip epochs that do not trigger logging
        if epoch < self.start_epoch or (epoch - self.start_epoch) % self.every_n_epochs != 0:
            return

        model.eval()

        # Unpack (X has multiple channels, SAR is target)
        x, sar = batch
        x = x.to(device)
        sar = sar.to(device)

        with torch.no_grad():
            pred = model(x, timestep=0).sample  # (B,1,H,W)

        # --- Select only IRWIN channel (channel 0) for visualization ---
        ir_only = x[:, 0:1, :, :]   # (B,1,H,W) keep as 4D tensor

        # ---- De-normalization ----
        if not isinstance(self.mean_X, (list, tuple, np.ndarray, dict)):
            mean = self.mean_X
            std = self.std_X
        else :
            mean = self.mean_X[0]
            std = self.std_X[0]

        ir_denorm   = self.denormalize(ir_only, mean , std)
        sar_denorm  = self.denormalize(sar, self.mean_sar, self.std_sar)
        pred_denorm = self.denormalize(pred, self.mean_sar, self.std_sar)

        # ---- Convert to numpy ----
        ir_np   = ir_denorm.squeeze(1).cpu().numpy()
        sar_np  = sar_denorm.squeeze(1).cpu().numpy()
        pred_np = pred_denorm.squeeze(1).cpu().numpy()

        num = min(self.num_samples, ir_np.shape[0])

        np.random.seed(0)
        for i in np.random.choice(len(ir_np), size=num, replace=False):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # IRWIN input (only channel 0)
            axes[0].imshow(ir_np[i], cmap=self.cmap_ir)
            axes[0].set_title("IRWIN (Channel 0)")
            axes[0].axis("off")

            # Real SAR
            sar_vis = np.where(self.mask[i] == 1, sar_np[i], np.nan)
            axes[1].imshow(sar_vis, cmap=self.cmap_sar)
            axes[1].set_title("Target SAR (knots)")
            axes[1].axis("off")

            # Predicted SAR
            pred_vis = np.where(self.mask[i] == 1, pred_np[i], np.nan)
            axes[2].imshow(pred_vis, cmap=self.cmap_sar)
            axes[2].set_title("Predicted SAR (knots)")
            axes[2].axis("off")

            if self.cyclone_ids is not None and self.sar_times is not None:
                fig.suptitle(f"Cyclone: {self.cyclone_ids[i]} — SAR Time: {self.sar_times[i]} — Epoch {epoch + 1}", fontsize=10)
            else:
                fig.suptitle(f"Epoch {epoch+1}", fontsize=14)

            plt.tight_layout()

            os.makedirs(os.path.join(Path(self.output_dir), "samples"), exist_ok=True)
            save_path = os.path.join(Path(self.output_dir), "samples", f"sample_{i}_epoch_{epoch + 1}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

            # Save distributions (unchanged)
            sar_sample = sar_denorm[i].squeeze(0)
            pred_sample = pred_denorm[i].squeeze(0)
            mask_valid = self.mask[i] == 1

            sar_valid = sar_sample[mask_valid]
            pred_valid = pred_sample[mask_valid]

            sar_plot = (sar_valid - sar_valid.min()) / (sar_valid.max() - sar_valid.min() + 1e-10)
            pred_plot = (pred_valid - pred_valid.min()) / (pred_valid.max() - pred_valid.min() + 1e-10)

            self.save_wind_sistribution(sar_valid, pred_valid)
            self.save_wind_sistribution(sar_plot, pred_plot, output="normalized")

            print(f"💾 Saved: {save_path}")

        # save distribution of wind speed of pred and true val to compare it
        #just in the lkast epoch
    def save_wind_sistribution(self, sar_denorm, pred_denorm, output= None):
        
            wind_true = sar_denorm.cpu().numpy().flatten()
            wind_pred = pred_denorm.cpu().numpy().flatten()
            min_value = min(wind_true.min(), wind_pred.min())
            plt.figure(figsize=(8,6))
        
            # plot the histograms
            plt.hist((wind_true ), bins=50, alpha=0.5, label='True SAR', color='blue', density=True)
            plt.hist((wind_pred ), bins=50, alpha=0.5, label='Predicted SAR', color='orange', density=True)
            plt.legend()
            plt.title("Distribution of Wind Speed: True vs Predicted SAR")
            plt.xlabel("Wind Speed (knots)")
            plt.ylabel("Density")
            plt.savefig(os.path.join(Path(self.output_dir ), f"wind_speed_distribution_{output}.png"), dpi=150)
            plt.close()

    def on_validation_plots(self, model, epoch, dataloader, device):
        print(f"📸 Logging validation samples at epoch {epoch +1}")

        all_ir = []
        all_sar = []

        # Parcourir tout le DataLoader et accumuler IR et SAR
        for ir, sar in dataloader:
            all_ir.append(ir)
            all_sar.append(sar)

        # Concaténer sur la dimension batch (dim=0)
        ir_full = torch.cat(all_ir, dim=0)   # → (Total, 1, H, W)
        sar_full = torch.cat(all_sar, dim=0) # → (Total, 1, H, W)

        # Créer un tuple exactement comme un batch
        batch_full = (ir_full, sar_full)

        self.log_batch(model, batch_full, epoch, device)

