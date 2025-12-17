import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import importlib
from matplotlib.patches import Circle
import src.IR_to_SAR.data_preparation.distribution_data_visualisation as distdata
importlib.reload(distdata)


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
        self.best_epoch = 0
        self.should_stop = False

    def on_validation_epoch_end(self, val_loss, epoch, **kwargs):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
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
             cmap_ir="gray", cmap_sar="viridis", num_epochs=0, infos=None, distance_km=None, mask_train=None, mask_val = None,target_dir = None,
             radial_mean=None):

        self.base_dir = Path(base_dir)
        # self.output_dir = self._create_unique_dir(self.base_dir)
        self.output_dir = target_dir

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
        self.mask_train = mask_train
        self.mask_val = mask_val
        self.num_epochs = num_epochs
        self.infos = infos
        self.radial_mean = radial_mean
        


    def _create_unique_dir(self, base_dir):
        i = 1
        while (base_dir / f"train_ir_sar_{i}").exists():
            i += 1
        new_dir = base_dir / f"train_ir_sar_{i-1}"
        # new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir
        
    def compute_vmax1d_rmax1d(self, sar_2d, cx, cy):
        """
        Compute Vmax1D and Rmax1D from a 2D SAR field.
        - Vmax1D: maximum wind speed along radial profile
        - Rmax1D: radius at which Vmax1D occurs
        """

        H, W = sar_2d.shape

        Y, X = np.indices((H, W))
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)

        Rmax = int(R.max())
        radii = np.arange(0, Rmax)

        vmax_profile = []

        for R0 in radii:
            mask = (np.abs(R - R0) < 0.5)

            if np.sum(mask) == 0:
                vmax_profile.append(np.nan)
            else:
                vmax_profile.append(np.nanmax(sar_2d[mask]))

        vmax_profile = np.array(vmax_profile)

        # Remove NaNs
        valid = ~np.isnan(vmax_profile)
        vmax_profile = vmax_profile[valid]
        radii = radii[valid]

        vmax1d = np.nanmax(vmax_profile)
        rmax1d = radii[np.nanargmax(vmax_profile)]

        # Convert:
        # - Vmax in knots
        # - Rmax in km (each pixel = 2 km)
        vmax1d_knots = vmax1d * 1.94384
        rmax1d_km = rmax1d

        return vmax1d_knots, rmax1d_km


    def log_batch(self, model, batch, epoch, device, set="validation"):
        """
        Log l'ensemble d'un batch : images + distributions pixel-par-pixel.
        Compatible avec un batch normal ou avec 'toute la validation concaténée'.
        """
        if epoch < self.start_epoch or (epoch - self.start_epoch) % self.every_n_epochs != 0:
            return

        model.eval()

        x, sar, mask, infos = batch
        cyclone_id = [d["cyclone_id"] for d in infos]             
        sar_time = [d["sar_time"] for d in infos] 
        vmax=  [d["vmax"] for d in infos] 
        analysis_vmax = [d["analysis_vmax"] for d in infos] 
        analysis_rmax = [d["analysis_rmax"] for d in infos] 
        analysis_center_quality_flag = [d["analysis_center_quality_flag"] for d in infos]  
                                                       
        x = x.to(device)
        sar = sar.to(device)
        mask = mask.to(device)
        
        # ------------------------------------------------------------
        # 1) Prédiction
        # ------------------------------------------------------------
        with torch.no_grad():
            pred = model(x, timestep=0).sample  # (B,1,H,W)

        # ------------------------------------------------------------
        # 2) Dé-normalisation
        # ------------------------------------------------------------

        def denorm(t, mean, std):
            return t * (std + 1e-10) + mean
            # return std + t*(1e-10 + mean  -std)

        def annular_denormalization(
            images_norm,
            stats,
            bin_size=1
        ):
            print(images_norm.shape)
            N, H, W = images_norm.shape
            cx, cy = H // 2, W // 2
            y, x = np.indices((H, W))
            radius = np.sqrt((y - cy)**2 + (x - cx)**2)
            radial_bins = (radius // bin_size).astype(np.int32)
            mean = stats["mean"]
            std = stats["std"]
            images = images_norm.copy()
            for b in range(len(mean)):
                images[:, radial_bins == b] = (
                    images[:, radial_bins == b] * std[b]
                ) + mean[b]
            return images
        
        def add_radial_mean(
            images_anom,
            radial_profile,
            bin_size=1
        ):

            single = images_anom.ndim == 2
            if single:
                images_anom = images_anom[None, ...]

            N, H, W = images_anom.shape
            cy, cx = H // 2, W // 2

            y, x = np.indices((H, W))
            radius = np.sqrt((y - cy)**2 + (x - cx)**2)
            radial_bins = (radius // bin_size).astype(np.int32)

            images = images_anom.copy()
            n_bins = len(radial_profile)

            for b in range(n_bins):
                images[:, radial_bins == b] += radial_profile[b]

            return images[0] if single else images




        # On ne visualise que le canal IRWIN (canal 0)
        ir = x[:, 0, :, :]
        ir = ir.squeeze().cpu().numpy()
        sar = sar.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()
        
        if self.norm == "z_score":
            ir_denorm   = denorm(ir,  self.mean_X[0],   self.std_X[0])
            sar_denorm  = denorm(sar, self.mean_sar,    self.std_sar)
            pred_denorm = denorm(pred, self.mean_sar,   self.std_sar)
        elif self.norm == "annular" :
            
            ir_denorm = annular_denormalization(ir,stats={"mean": self.mean_X[0],"std":  self.std_X[0]})
            sar_denorm = annular_denormalization(sar, stats={"mean": self.mean_sar,"std":  self.std_sar} )
            pred_denorm = annular_denormalization(pred, stats={"mean": self.mean_sar,"std":  self.std_sar})

        # Conversion numpy
        ir_np   = ir_denorm
        sar_np  = sar_denorm
        pred_np = pred_denorm
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()

        batch_size = ir_np.shape[0]
        num = min(self.num_samples, batch_size)

        # Tirage aléatoire d'exemples
        np.random.seed(0)
        sample_ids = np.random.choice(batch_size, size=num, replace=False)

        # ------------------------------------------------------------
        # 3) Plots par échantillon
        # ------------------------------------------------------------
        os.makedirs(os.path.join(self.output_dir, "samples", set), exist_ok=True)
        # Convert to numpy
        sar_all  = sar_denorm.flatten()
        pred_all = pred_denorm.flatten()
        mask_all = mask_np.flatten()

        valid = mask_all == 1

        sar_valid  = sar_all[valid]
        pred_valid = pred_all[valid]

        # Conversion en knots
        sar_knots_flat  = sar_valid * 1.94384
        pred_knots_flat = pred_valid * 1.94384

        # → Seulement pour la distribution
        distdata.compare_sar_distribution(
            sar_knots_flat,
            pred_knots_flat,
            self.output_dir,
            set=set,
            epoch=epoch
        )

        # =================================================
        # 2) Vmax & radial → PAS de flatten (2D + masque)
        # =================================================
        sar_2d  = sar_denorm   # (B, H, W)
        pred_2d = pred_denorm # (B, H, W)
        mask_2d = mask_np                                # (B, H, W)

        # Conversion en knots
        sar_2d_knots  = sar_2d * 1.94384
        pred_2d_knots = pred_2d * 1.94384

        # → Vmax utilise les champs 2D
        distdata.vmax_compare(
            analysis_vmax,
            pred_2d_knots,
            self.output_dir,
            set=set,
            epoch=epoch
        )

        distdata.rmax_compare(
            analysis_rmax,
            pred_2d_knots,
            self.output_dir,
            set=set,
            epoch=epoch
        )

        # → Radial Vmax utilise aussi les champs 2D
        distdata.compare_radial_vmax(
            sar_2d_knots,
            pred_2d_knots,
            output_dir=self.output_dir,
            set=set,
            epoch=epoch,
            plot=False
        )

        # mae 
        distdata.compute_mae_metric(
            sar_2d_knots,
            pred_2d_knots,
            output_dir=self.output_dir,
            set=set,
            epoch=epoch,
            plot=False
        )
       
        for i in sample_ids:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            H, W = sar_np[i].shape
            cx, cy = W // 2, H // 2

            # Compute radial metrics
            
            vmax1d_pred, rmax1d_pred = self.compute_vmax1d_rmax1d(pred_np[i], cx, cy)

            
            rmax = analysis_rmax[i] / 2000    #pixel
            vmax = analysis_vmax[i] *1.94384
            if vmax is None or np.isnan(rmax):
                vmax = 99999
            if rmax is None or np.isnan(rmax):
                rmax = 99999

            # IRWIN
            axes[0].imshow(ir_np[i], cmap=self.cmap_ir)
            axes[0].set_title("IRWIN (Channel 0)")
            axes[0].axis("off")

            # === TRUE SAR ===
            sar_vis = np.where(mask_np[i]==1, sar_np[i], np.nan)
            axes[1].imshow(sar_vis, cmap=self.cmap_sar)
            axes[1].set_title("True SAR (knots)")
            axes[1].axhline(y=cy, color="black", linewidth=1)
            axes[1].axvline(x=cx, color="black", linewidth=1)

            if rmax is not None:
                axes[1].add_patch(Circle((cx, cy), radius=rmax, color="black", fill=False, linestyle="--"))

            axes[1].axis("off")

            # === PREDICTED SAR ===
            # pred_vis = np.where(mask_np[i]==1, pred_np[i], np.nan)
            axes[2].imshow(pred_np[i], cmap=self.cmap_sar)
            axes[2].set_title("Predicted SAR (knots)")
            axes[2].axhline(y=cy, color="black", linewidth=1)
            axes[2].axvline(x=cx, color="black", linewidth=1)

            if rmax is not None:
                axes[2].add_patch(Circle((cx, cy), radius=rmax, color="black", fill=False, linestyle="--"))

            axes[2].axis("off")

            # === TITLE including Rmax1D and Vmax1D ===
            fig.suptitle(
                f"Cyclone: {cyclone_id[i]} — SAR Time: {sar_time[i]} — Epoch {epoch+1}\n"
                f"Analysis Rmax = {rmax*2:.1f} Km — Predicted Rmax1D = {rmax1d_pred:.1f} Km\n"
                f"Anaysis Vmax = {vmax:.1f} kt — Predicted Vmax1D = {vmax1d_pred:.1f} kt",
                fontsize=10
            )


            plt.tight_layout()
            save_path = os.path.join(self.output_dir, "samples",set,  f"sample_{i}_epoch_{epoch+1}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

        print(f"💾 Saved {num} sample images.")

        
        


        # save distribution of wind speed of pred and true val to compare it
        #just in the lkast epoch

    def on_validation_plots(self, model, epoch, dataloader, device):
        print(f"📸 Logging validation samples at epoch {epoch +1}")

        all_ir_val = []
        all_sar_val = []
        all_ir_train = []
        all_sar_train = []

        mask_train = []
        mask_val = []
        infos_val = []
        infos_train = []

        # Parcourir tout le DataLoader et accumuler IR et SAR
        for ir, sar, mask, inf in dataloader[1]:
            all_ir_val.append(ir)
            all_sar_val.append(sar)
            mask_val.append(mask)
            infos_val.append(inf)
    
        
        for ir, sar, mask, inf in dataloader[0]:
            all_ir_train.append(ir)
            all_sar_train.append(sar)
            mask_train.append(mask)
            infos_train.append(inf)

      

        # Concaténer sur la dimension batch (dim=0)
        ir_full_val = torch.cat(all_ir_val, dim=0)   # → (Total, 1, H, W)
        sar_full_val = torch.cat(all_sar_val, dim=0) # → (Total, 1, H, W)
        mask_val = torch.cat(mask_val, dim=0)
        infos_val = [d for batch in infos_val for d in batch]   # concat lists
        

        ir_full_train = torch.cat(all_ir_train, dim=0)   # → (Total, 1, H, W)
        sar_full_train = torch.cat(all_sar_train, dim=0) # → (Total, 1, H, W)
        mask_train = torch.cat(mask_train, dim=0)
        infos_train = [d for batch in infos_train for d in batch]   # concat lists
        

        # Créer un tuple exactement comme un batch
        batch_full_val = (ir_full_val, sar_full_val, mask_val, infos_val)
        batch_full_train = (ir_full_train, sar_full_train, mask_train, infos_train)

        self.log_batch(model, batch_full_val, epoch, device)
        self.log_batch(model, batch_full_train, epoch, device, set="train")

