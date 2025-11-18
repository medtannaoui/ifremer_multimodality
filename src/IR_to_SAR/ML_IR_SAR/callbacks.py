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
    Callback to log example predictions for IR → SAR regression.

    For a few validation samples, it plots:
        - Input IR
        - Predicted SAR
        - Target SAR

    Shapes:
        IR batch   : (B, 1, H, W)
        SAR batch  : (B, 1, H, W)
    """

    def __init__(
        self,
        output_dir: str,
        num_samples: int = 4,
        vmax_ir: float | None = None,
        vmax_sar: float | None = None,
    ):
        """
        Args:
            output_dir (str): directory where plots will be saved.
            num_samples (int): how many samples to plot from a batch.
            vmax_ir (float or None): fixed max color scale for IR (if None → auto).
            vmax_sar (float or None): fixed max color scale for SAR (if None → auto).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.vmax_ir = vmax_ir
        self.vmax_sar = vmax_sar

    def log_batch(
        self,
        model: torch.nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        epoch: int,
        device: torch.device | str = "cuda",
    ):
        """
        Generate and save a figure with IR input, predicted SAR, and target SAR.

        Args:
            model : trained model (in eval mode will be set internally).
            batch : tuple (ir, sar) from DataLoader.
                ir  shape: (B, 1, H, W)
                sar shape: (B, 1, H, W)
            epoch : current epoch index.
            device: model/device to use ("cuda" or "cpu").
        """
        model.eval()
        ir, sar = batch
        ir = ir.to(device)
        sar = sar.to(device)

        with torch.no_grad():
            pred = model(ir)  # (B, 1, H, W)

        ir_np   = ir.squeeze(1).cpu().numpy()    # (B, H, W)
        sar_np  = sar.squeeze(1).cpu().numpy()   # (B, H, W)
        pred_np = pred.squeeze(1).cpu().numpy()  # (B, H, W)

        num = min(self.num_samples, ir_np.shape[0])

        fig, axes = plt.subplots(num, 3, figsize=(12, 4 * num))
        if num == 1:
            axes = np.expand_dims(axes, axis=0)

        fig.suptitle(f"IR → SAR validation samples (Epoch {epoch})", fontsize=16)

        for i in range(num):
            # IR input
            ax_in = axes[i, 0]
            vmin_ir = np.nanmin(ir_np[i])
            vmax_ir = self.vmax_ir if self.vmax_ir is not None else np.nanmax(ir_np[i])
            im_in = ax_in.imshow(ir_np[i], cmap="gray", vmin=vmin_ir, vmax=vmax_ir)
            ax_in.set_title(f"Input IR - Sample {i}")
            ax_in.axis("off")
            fig.colorbar(im_in, ax=ax_in, orientation="horizontal", pad=0.15)

            # Predicted SAR
            ax_pred = axes[i, 1]
            vmin_sar = np.nanmin([pred_np[i], sar_np[i]])
            vmax_sar = (
                self.vmax_sar
                if self.vmax_sar is not None
                else np.nanmax([pred_np[i], sar_np[i]])
            )
            im_pred = ax_pred.imshow(pred_np[i], cmap="viridis", vmin=vmin_sar, vmax=vmax_sar)
            ax_pred.set_title("Predicted SAR")
            ax_pred.axis("off")
            fig.colorbar(im_pred, ax=ax_pred, orientation="horizontal", pad=0.15)

            # Target SAR
            ax_tar = axes[i, 2]
            im_tar = ax_tar.imshow(sar_np[i], cmap="viridis", vmin=vmin_sar, vmax=vmax_sar)
            ax_tar.set_title("Target SAR")
            ax_tar.axis("off")
            fig.colorbar(im_tar, ax=ax_tar, orientation="horizontal", pad=0.15)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = self.output_dir / f"val_samples_epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"📊 Saved validation samples plot to {save_path}")
