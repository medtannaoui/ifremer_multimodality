import os
from pathlib import Path
import pickle as pkl
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
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
             cmap_ir="gray", cmap_sar="viridis", num_epochs=0, infos_train=None, infos_val=None, infos_test=None, 
             mask_train=None, mask_val = None,mask_test=None,
             target_dir = None,input_data="normal", output_data="sar"
             ):

        self.base_dir = Path(base_dir)
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
        self.mask_test=mask_test
        self.num_epochs = num_epochs
        self.infos_train = infos_train
        self.infos_val = infos_val
        self.infos_test = infos_test

        self.vmax_bins_knots = None
        self.rmax_bins_km = None

        self.input_data = input_data
        self.output_data=output_data

        
        
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
        vmax1d_knots = vmax1d * 1.94384   #kt
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
        
        with torch.no_grad():
            pred = model(x, timestep=0).sample  # (B,1,H,W)
            

        def denorm(t, mean, std):
            return t * (std + 1e-10) + mean

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

        def moment_to_sar(moment):
            assert moment.ndim == 3, "moment must be (N, H, W)"

            N, H, W = moment.shape

            y, x = np.indices((H, W))
            cy, cx = H // 2, W // 2

            r = np.sqrt((x - cx)**2 + (y - cy)**2)

            r_safe = np.maximum(r, 1.0)

            sar = moment / r_safe[None, :, :]

            return sar

        def _compute_stats(err):
            """err: 1D numpy array (no nan)"""
            bias = np.mean(err)
            std  = np.std(err)
            rmse = np.sqrt(np.mean(err**2))
            mae  = np.mean(np.abs(err))
            return bias, std, rmse, mae
        

        def _split_bins_from_train( values_train, n_intervals=3):
            """Build bins from TRAIN only using linspace(min,max,n_intervals+1)."""
            v = np.asarray(values_train)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return None
            bins = np.linspace(v.min(), v.max(), n_intervals + 1)
            return bins
        

        def _plot_4panel_error_hist(errors, cat_values, bins, title_prefix, unit, save_path, xlim=None):
            """
            errors: 1D array (pred - analysis) in desired unit
            cat_values: 1D array used to assign categories (typically analysis values)
            bins: array length 4 -> 3 intervals
            """
            errors = np.asarray(errors)
            cat_values = np.asarray(cat_values)

            # keep finite
            ok = np.isfinite(errors) & np.isfinite(cat_values)
            errors = errors[ok]
            cat_values = cat_values[ok]

            fig, axes = plt.subplots(1, 4, figsize=(18, 5))

            def draw(ax, err_subset, subtitle):
                err_subset = np.asarray(err_subset)
                err_subset = err_subset[np.isfinite(err_subset)]
                if err_subset.size == 0:
                    ax.set_title(subtitle + "\n(empty)")
                    ax.grid(True, linestyle="--", alpha=0.4)
                    return

                bias, std, rmse, mae = _compute_stats(err_subset)

                ax.hist(err_subset, bins=40)
                ax.set_title(subtitle)
                ax.set_xlabel(f"Error ({unit})")
                ax.set_ylabel("Count")
                ax.grid(True, linestyle="--", alpha=0.4)

                txt = (f"bias = {bias:.2f} {unit}\n"
                        f"stddev = {std:.2f} {unit}\n"
                        f"rmse = {rmse:.2f} {unit}\n"
                        f"mae = {mae:.2f} {unit}\n"
                        f"n = {err_subset.size}\n")
                ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                        ha="right", va="top", fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

                if xlim is not None:
                    ax.set_xlim(xlim)

            # Panel 1: all
            draw(axes[0], errors, f"{title_prefix}\nAll cases")

            # Panels 2-4: three categories from bins
            # categories: [bins[0], bins[1]), [bins[1], bins[2]), [bins[2], bins[3]]
            for k in range(3):
                lo, hi = bins[k], bins[k+1]
                if k < 2:
                    sel = (cat_values >= lo) & (cat_values < hi)
                else:
                    sel = (cat_values >= lo) & (cat_values <= hi)

                subtitle = f"{title_prefix}\nCat{k+1}: [{lo:.1f}, {hi:.1f}]"
                draw(axes[k+1], errors[sel], subtitle)

            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            plt.close(fig)


        # ---------------------------------------------------------------------------------------------------------------------------------------------------
        # On ne visualise que le canal IRWIN (canal 0)
        ch = 4 if x.shape[1] == 9 else 0
        ir = x[:, ch, :, :]
        ir = ir.squeeze().cpu().numpy()
        sar = sar.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()
        
        ir_denorm   = denorm(ir,  self.mean_X[ch],   self.std_X[ch])
       

        if self.norm == "z_score":
            sar_denorm  = denorm(sar, self.mean_sar,    self.std_sar)
            pred_denorm = denorm(pred, self.mean_sar,   self.std_sar)

        elif self.norm == "annular" :
            sar_denorm = annular_denormalization(sar, stats={"mean": self.mean_sar,"std":  self.std_sar} )
            pred_denorm = annular_denormalization(pred, stats={"mean": self.mean_sar,"std":  self.std_sar})

        if self.output_data == "aam":
            sar_denorm = moment_to_sar(sar_denorm)
            pred_denorm = moment_to_sar(pred_denorm)

        # Conversion numpy
        ir_np   = ir_denorm
        sar_np  = sar_denorm
        pred_np = pred_denorm
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np=mask
        if isinstance(infos, torch.Tensor):
            infos_np = infos.cpu().numpy()
        else : 
            infos_np=infos
        
        B, H, W = pred_np.shape
        cx, cy = W // 2, H // 2

        pred_vmax_knots = np.full(B, np.nan, dtype=np.float32)
        pred_rmax_km    = np.full(B, np.nan, dtype=np.float32)

        for i in range(B):
            vmax1d_pred_kt, rmax1d_pred_pix = self.compute_vmax1d_rmax1d(pred_np[i], cx, cy)
            pred_vmax_knots[i] = vmax1d_pred_kt
            pred_rmax_km[i] = rmax1d_pred_pix * 2.0  # 2 km per pixel

        # ==========================
        # Analysis values in same units
        # ==========================
        analysis_vmax = np.array(analysis_vmax, dtype=np.float32)  # m/s (often)
        analysis_rmax = np.array(analysis_rmax, dtype=np.float32)  # meters

        analysis_vmax_knots = analysis_vmax * 1.94384
        analysis_rmax_km = analysis_rmax / 1000.0

        # filter missing analysis
        ok_vmax = np.isfinite(analysis_vmax_knots) & np.isfinite(pred_vmax_knots)
        ok_rmax = np.isfinite(analysis_rmax_km) & np.isfinite(pred_rmax_km)

        # Errors
        err_vmax = (pred_vmax_knots - analysis_vmax_knots)[ok_vmax]
        cat_vmax = analysis_vmax_knots[ok_vmax]  # categories based on analysis
        err_rmax = (pred_rmax_km - analysis_rmax_km)[ok_rmax]
        cat_rmax = analysis_rmax_km[ok_rmax]     # categories based on analysis

        # ==========================
        # Build bins from TRAIN only (once), reuse for val/test
        # ==========================
        if set == "train":
            self.vmax_bins_knots = _split_bins_from_train(values_train=cat_vmax, n_intervals=3)
            self.rmax_bins_km    = _split_bins_from_train(values_train=cat_rmax, n_intervals=3)

        # fallback if train not logged yet
        if self.vmax_bins_knots is None:
            self.vmax_bins_knots = _split_bins_from_train(values_train=cat_vmax, n_intervals=3)
        if self.rmax_bins_km is None:
            self.rmax_bins_km = _split_bins_from_train(values_train=cat_rmax, n_intervals=3)

        # If still None, skip plotting
        if self.vmax_bins_knots is not None and err_vmax.size > 0:
            save_path_vmax = os.path.join(self.output_dir, "errors_hist", set, f"vmax_error_epoch_{epoch+1:04d}.png")
            _plot_4panel_error_hist(
                errors=err_vmax,
                cat_values=cat_vmax,
                bins=self.vmax_bins_knots,
                title_prefix="Vmax error",
                unit="kt",
                save_path=save_path_vmax,
                xlim=None  # you can set e.g. (-80, 80)
            )

        if self.rmax_bins_km is not None and err_rmax.size > 0:
            save_path_rmax = os.path.join(self.output_dir, "errors_hist", set, f"rmax_error_epoch_{epoch+1:04d}.png")
            _plot_4panel_error_hist(
                errors=err_rmax,
                cat_values=cat_rmax,
                bins=self.rmax_bins_km,
                title_prefix="Rmax error",
                unit="km",
                save_path=save_path_rmax,
                xlim=None  # you can set e.g. (-120, 120)
            )

        batch_size = ir_np.shape[0]
        num = batch_size if batch_size< self.num_samples else self.num_samples

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
        mask_2d = mask_np 
        os.makedirs(os.path.join(self.output_dir,'predictions_denormalisees',set),exist_ok=True)                               # (B, H, W)
        with open(os.path.join(self.output_dir,"predictions_denormalisees",set,"predictions_denormalisées.pkl"),"wb") as f:
            pkl.dump({f"{set}":[ir_np,sar_2d,pred_2d,mask_2d,infos_np]},f)

        # Conversion en knots
        sar_2d_knots  = sar_2d * 1.94384
        pred_2d_knots = pred_2d * 1.94384



        # → Vmax utilise les champs 2D
        for min,max in zip([19,63,83,96,113],[63,83,96,113,200]):
            distdata.vmax_compare(
                analysis_vmax,
                pred_2d_knots,
                self.output_dir,
                set=set,
                epoch=epoch,
                min=min,
                max=max
            )
        distdata.vmax_compare(
                analysis_vmax,
                pred_2d_knots,
                self.output_dir,
                set=set,
                epoch=epoch
            )
        

        for min,max in zip([0,30,60],[30,60,100]):
            distdata.rmax_compare(
                analysis_rmax,
                pred_2d_knots,
                self.output_dir,
                set=set,
                epoch=epoch,
                min=min,
                max=max
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
            vmax = analysis_vmax[i] *1.94384  #m/s to kt
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
    
    def on_validation_plots(self, model, epoch, dataloader, device):
            print(f"📸 Logging validation samples at epoch {epoch +1}")
            print("size of dataloader",len(dataloader))

            all_ir_val = []
            all_sar_val = []
            all_sar_test = []
            all_ir_test = []
            all_ir_train = []
            all_sar_train = []
            ir_anggrek, sar_anggrek = [], []


            mask_train = []
            mask_val = []
            mask_test = []
            mask_anggrek  = []
            infos_val = []
            infos_train = []
            infos_test = []
            infos_anggrek = []

            for ir, sar, mask, inf in dataloader[1]:
                all_ir_val.append(ir)
                all_sar_val.append(sar)
                mask_val.append(mask)
                infos_val.append(inf)
            # train
            for ir, sar, mask, inf in dataloader[0]:
                if inf["augmentation"]==1:
                    continue
                all_ir_train.append(ir)
                all_sar_train.append(sar)
                mask_train.append(mask)
                infos_train.append(inf)
            
            for ir, sar, mask, inf in dataloader[2]:
                all_ir_test.append(ir)
                all_sar_test.append(sar)
                mask_test.append(mask)
                infos_test.append(inf)
            
            #anggrek data
            for ir, sar, mask, inf in dataloader[3]:
                ir_anggrek.append(ir)
                sar_anggrek.append(sar)
                mask_anggrek.append(mask)
                infos_anggrek.append(inf)

        

            # Concaténer sur la dimension batch (dim=0)
            ir_full_val = torch.cat(all_ir_val, dim=0)   # → (Total, 1, H, W)
            sar_full_val = torch.cat(all_sar_val, dim=0) # → (Total, 1, H, W)
            mask_val = torch.cat(mask_val, dim=0)
            infos_val = [d for batch in infos_val for d in batch]   # concat lists
            

            ir_full_train = torch.cat(all_ir_train, dim=0)   # → (Total, 1, H, W)
            sar_full_train = torch.cat(all_sar_train, dim=0) # → (Total, 1, H, W)
            mask_train = torch.cat(mask_train, dim=0)
            infos_train = [d for batch in infos_train for d in batch]  # concat lists
            
            ir_full_test = torch.cat(all_ir_test, dim=0)   # → (Total, 1, H, W)
            sar_full_test = torch.cat(all_sar_test, dim=0) # → (Total, 1, H, W)
            mask_test = torch.cat(mask_test, dim=0)
            infos_test = [d for batch in infos_test for d in batch]  

            ir_full_anggrek = torch.cat(ir_anggrek, dim=0)
            sar_full_anggrek = torch.cat(sar_anggrek, dim=0)
            mask_anggrek = torch.cat(mask_anggrek, dim=0)
            infos_anggrek = [d for batch in infos_anggrek for d in batch]

            # Créer un tuple exactement comme un batch
            batch_full_val = (ir_full_val, sar_full_val, mask_val, infos_val)
            batch_full_train = (ir_full_train, sar_full_train, mask_train, infos_train)
            batch_full_test = (ir_full_test, sar_full_test, mask_test, infos_test)
            batch_full_anggrek = (ir_full_anggrek, sar_full_anggrek, mask_anggrek, infos_anggrek)

            self.log_batch(model, batch_full_val, epoch, device)
            self.log_batch(model, batch_full_train, epoch, device, set="train")
            self.log_batch(model, batch_full_test, epoch, device, set="test")
            self.anggrek_plots(model, batch_full_anggrek, epoch, device)



        
        


        # save distribution of wind speed of pred and true val to compare it
        #just in the lkast epoch

    

    def anggrek_plots(self, model, batch, epoch, device):
        """
        Produces ONLY:
        (1) field_plots/<YYYYmmddHHMMSS>/<YYYYmmddHHMMSS>_fields.png with IR + PRED
        (2) vmax_comparison.png : Vmax (infos) vs Vmax_pred (max of pred_den), with RMSE
        """
        import os
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import torch

        model.eval()

        # -------------------------
        # Unpack batch
        # -------------------------
        x, _, _, infos = batch
        x = x.to(device)

        # -------------------------
        # Predict
        # -------------------------
        with torch.no_grad():
            pred = model(x, timestep=0).sample  # (B,1,H,W)

        # -------------------------
        # Helpers
        # -------------------------
        def denorm(t, mean, std):
            return t * (std + 1e-10) + mean

        def annular_denormalization(images_norm, stats, bin_size=1):
            # images_norm: (B,H,W)
            N, H, W = images_norm.shape
            cx, cy = (W - 1) / 2, (H - 1) / 2
            y, x_ = np.indices((H, W))
            radius = np.sqrt((y - cy) ** 2 + (x_ - cx) ** 2)
            radial_bins = (radius // bin_size).astype(np.int32)
            mean = stats["mean"]
            std = stats["std"]
            images = images_norm.copy()
            for b in range(len(mean)):
                images[:, radial_bins == b] = images[:, radial_bins == b] * std[b] + mean[b]
            return images

        def moment_to_sar(moment):
            # moment: (B,H,W)
            assert moment.ndim == 3, "moment must be (B,H,W)"
            B, H, W = moment.shape
            y, x = np.indices((H, W))
            cy, cx = H // 2, W // 2
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            r_safe = np.maximum(r, 1.0)
            return moment / r_safe[None, :, :]

        def robust_limits(arr, qmin=1, qmax=99):
            a = np.asarray(arr)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return None, None
            return np.percentile(a, qmin), np.percentile(a, qmax)

        # -------------------------
        # To numpy + denorm
        # -------------------------
        x_np = x.detach().cpu().numpy()  # (B,C,H,W)
        pred_np = pred.detach().squeeze().cpu().numpy()  # (B,H,W) or (H,W) if B=1

        if pred_np.ndim == 2:
            pred_np = pred_np[None, ...]  # force (B,H,W)

        B = x_np.shape[0]
        ch = 0  # IR channel index
        ir = x_np[:, ch, :, :]  # (B,H,W)
        pred1 = pred_np          # (B,H,W)

        ir_den = denorm(ir, self.mean_X[ch], self.std_X[ch])

        if self.norm == "z_score":
            pred_den = denorm(pred1, self.mean_sar, self.std_sar)
        elif self.norm == "annular":
            pred_den = annular_denormalization(pred1, stats={"mean": self.mean_sar, "std": self.std_sar})
        else:
            # fallback: assume already denorm
            pred_den = pred1

        if getattr(self, "output_data", "") == "aam":
            pred_den = moment_to_sar(pred_den)

        # -------------------------
        # Sort by time
        # -------------------------
        sar_time = [d.get("date") for d in infos]
        time_parsed = pd.to_datetime(sar_time, errors="coerce")
        order = np.argsort(time_parsed.values.astype("datetime64[ns]"))

        ir_den = ir_den[order]
        pred_den = pred_den[order]
        infos_ord = [infos[i] for i in order]
        time_parsed = time_parsed[order]

        

        # -------------------------
        # Output dirs
        # -------------------------
        out_root = Path(self.output_dir) / "anggrek_monitoring"
        field_dir = out_root / "field_plots"
        field_dir.mkdir(parents=True, exist_ok=True)


        # -------------------------
        # (1) Save IR + Pred per date
        # -------------------------
        for i in range(B):
            t = time_parsed[i]

            if pd.isna(t):
                date_key = f"unknown_{i:03d}"
                fname = f"unknown_{i:03d}_fields.png"
                supt = f"Unknown time (idx={i})"
            else:
                date_key = t.strftime("%Y%m%d%H%M%S")  
                fname = f"{date_key}_fields.png"
                supt = t.strftime("%Y-%m-%d %H:%M:%S")

            sub = field_dir
            sub.mkdir(parents=True, exist_ok=True)

            fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

            im0 = axs[0].imshow(ir_den[i], cmap=self.cmap_ir)
            axs[0].set_title("IRWIN")
            axs[0].axis("off")
            

            im1 = axs[1].imshow(pred_den[i], cmap=self.cmap_sar)
            axs[1].set_title("Prediction")
            axs[1].axis("off")
            

            fig.suptitle(supt)
            fig.savefig(sub / fname, dpi=150)
            plt.close(fig)

        #Resample SAr from 2km to 3km
        @torch.no_grad()
        def resample_2km_to_3km_torch(xs):
            resamples = []

            for x in xs:
                # numpy -> torch (CPU)
                x = torch.as_tensor(x, dtype=torch.float32, device="cpu")

                # HxW -> 1x1xHxW
                if x.dim() == 2:
                    x = x.unsqueeze(0).unsqueeze(0)

                H, W = x.shape[-2:]
                new_H = int(H * 2 / 3)
                new_W = int(W * 2 / 3)

                # CPU interpolation
                x3 = F.interpolate(x, size=(new_H, new_W), mode="bilinear", align_corners=False)

                # back to numpy (HxW)
                resamples.append(x3.squeeze(0).squeeze(0).cpu().numpy())

            return np.stack(resamples, axis=0)
        
        pred_denorm_3m = resample_2km_to_3km_torch(pred_den)
        # -------------------------
        # (2) Vmax comparison plot
        # -------------------------
        # truth vmax from infos (m/s)
        vmax = np.array([d.get("vmax", np.nan) for d in infos_ord], dtype=float)

        # predicted vmax from pred field (m/s)
        pred_vmax = np.nanmax(pred_denorm_3m.reshape(B, -1), axis=1)

        ok = np.isfinite(vmax) & np.isfinite(pred_vmax)
        rmse = float(np.sqrt(np.mean((pred_vmax[ok] - vmax[ok]) ** 2))) if np.any(ok) else np.nan

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(time_parsed, vmax, color="black", linewidth=2, label="Vmax")
        ax.plot(time_parsed, pred_vmax, color="magenta", linewidth=2, label="Pred Vmax")

        ax.set_title(f"Lifecycle Vmax Comparison — RMSE = {rmse:.2f} m/s")
        ax.set_xlabel("Time")
        ax.set_ylabel("Vmax (m/s)")

        # clean date axis like your IBTrACS figure
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_root / "vmax_comparison.png", dpi=150)
        plt.close(fig)





        