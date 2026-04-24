import os
from pathlib import Path
import pickle as pkl
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch.nn.functional as F
import numpy as np
import importlib
from matplotlib.patches import Circle
import src.IR_to_SAR.data_preparation.distribution_data_visualisation as distdata
importlib.reload(distdata)



import src.IR_to_SAR.ML_IR_SAR.flow_matching_inference as fm_inf
importlib.reload(fm_inf)
from src.IR_to_SAR.ML_IR_SAR import flow_matching_inference as fm_inf



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



def _euler_ode(model, z, ir_input, num_steps, device):
            """Euler ODE solver (no-grad context assumed)."""
            x_t = z
            dt  = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((x_t.shape[0],), i / num_steps, device=device)
                model_input   = torch.cat([x_t, ir_input], dim=1)
                pred_velocity = model(model_input, t).sample
                x_t = x_t + pred_velocity * dt
            return x_t

class LogValidationSamples:
    """
    Callback to log IR → SAR predictions after epoch 20,
    saving per-sample plots in a unique train directory.
    """

    def __init__(self, base_dir, mean_X, std_X, mean_sar, std_sar,
             cmap_ir="gray", cmap_sar="viridis", num_epochs=0, 
             infos_train=None, infos_val=None, infos_test=None, 
             mask_train=None, mask_val = None,mask_test=None,target_dir=None,
             cfg=None
             ):

        self.base_dir = Path(base_dir)
        self.output_dir = target_dir

        self.num_samples = cfg.num_val_exemples
        self.start_epoch = cfg.start_epoch
        self.every_n_epochs = cfg.plot_interval  
        self.cmap_ir = cmap_ir
        self.cmap_sar = cmap_sar

        self.mean_sar = mean_sar
        self.mean_X = mean_X
        self.std_sar = std_sar
        self.std_X = std_X
        
        self.norm = cfg.norm
        self.mask_train = mask_train
        self.mask_val = mask_val
        self.mask_test=mask_test
        self.num_epochs = num_epochs
        self.infos_train = infos_train
        self.infos_val = infos_val
        self.infos_test = infos_test

        self.vmax_bins_knots = None
        self.rmax_bins_km = None

        self.input_data = cfg.input_data
        self.output_data=cfg.output_data

        
        self.conditional_model = cfg.conditional_model
        self.log_wind = cfg.log_wind
        self.crop_sar = cfg.crop_sar

        self.irwin_channels = cfg.irwin_channels
        self.regrid_ir = cfg.regrid_ir
        self.add_era5 = cfg.add_era5
        self.cfg = cfg
        
        
    def _create_unique_dir(self, base_dir):
        i = 1
        while (base_dir / f"train_ir_sar_{i}").exists():
            i += 1
        new_dir = base_dir / f"train_ir_sar_{i-1}"
        # new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir
        
    def compute_vmax1d_rmax1d(self, sar_2d):
        """
        Compute Vmax1D and Rmax1D from a 2D SAR field.
        - Vmax1D: maximum wind speed along radial profile
        - Rmax1D: radius at which Vmax1D occurs
        """
        H, W = sar_2d.shape
        cx, cy = W//2, H//2
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
        vmax1d_knots = vmax1d
        rmax1d_km = rmax1d * 2
        return vmax1d_knots, rmax1d_km
    

    def log_batch(self, model, batch, epoch, device, set="validation",ode_pred=None,reg_model=None,resid_stats=None):
        """
        Log l'ensemble d'un batch : images + distributions pixel-par-pixel.
        Compatible avec un batch normal ou avec 'toute la validation concaténée'.
        """
        

        model.eval()
        if ode_pred is None:
            x_fm, sar_fm, mask_fm, infos_fm = batch
        x, sar, mask, infos, ode_pred = batch if ode_pred is not None else (*batch, None)
        
        cyclone_id = [d["cyclone_id"] for d in infos]             
        sar_time = [d["sar_time"] for d in infos] 
        vmax=  [d["vmax"] for d in infos] 
        analysis_vmax = [d["analysis_vmax"] for d in infos] 
        analysis_rmax = [d["analysis_rmax"] for d in infos] 
        analysis_center_quality_flag = [d["analysis_center_quality_flag"] for d in infos]  
                                                       
        x = x.to(device)
        sar = sar.to(device)
        mask = mask.to(device)
        

        if ode_pred is None:
            with torch.no_grad():
                if not self.conditional_model:
                    if not self.cfg.use_flow_matching:
                        pred = model(x, timestep=0).sample
                    elif not self.cfg.use_residu:
                        B = x.shape[0]
                        H = x.shape[2]
                        W = x.shape[3]
                        z = torch.randn(B, 1, H, W, device=device)
                        pred = _euler_ode(model, z, x, self.cfg.fm_num_inference_steps, device)
                    else:
                        B, _, H, W = x.shape
                        z = torch.randn(B, 1, H, W, device=device)

                        mean_pred = reg_model(x, timestep=0).sample

                        resid_norm = fm_inf.ode_solver_residual(
                            fm_model=model,
                            z=z,
                            mean_pred=mean_pred,
                            num_steps=self.cfg.fm_num_inference_steps
                        )

                        pred = fm_inf.reconstruct_from_residual(
                            resid_norm,
                            mean_pred,
                            resid_stats["mean"],
                            resid_stats["std"]
                        )

                else:
                    shear = torch.stack([
                        torch.as_tensor(d["shear"], dtype=torch.float32)
                        for d in infos
                    ]).to(device)
                    pred = model(x, timestep=0, cond=shear).sample
        
        else : 
            pred = ode_pred.to(device)
            print("no ode pred",pred.shape)
            

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

        def _split_bins_from_train( values_train, n_intervals=3):
            """Build bins from TRAIN only using linspace(min,max,n_intervals+1)."""
            v = np.asarray(values_train)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return None
            bins = np.linspace(v.min(), v.max(), n_intervals + 1)
            return bins
        

        def _compute_stats(err):
            """err: 1D numpy array (no nan)"""
            bias = np.mean(err)
            std  = np.std(err)
            rmse = np.sqrt(np.mean(err**2))
            mae  = np.mean(np.abs(err))
            return bias, std, rmse, mae


        def _plot_4panel_error_hist(errors, cat_values, bins, title_prefix, unit, save_path, xlim=None):
            """
            errors: 1D array (pred - analysis) in desired unit
            cat_values: 1D array used to assign categories
            bins: array length 4 -> 3 intervals
            """
            errors = np.asarray(errors)
            cat_values = np.asarray(cat_values)

            ok = np.isfinite(errors) & np.isfinite(cat_values)
            errors = errors[ok]
            cat_values = cat_values[ok]

            fig, axes = plt.subplots(1, 4, figsize=(18, 5))

            def draw(ax, err_subset, cat_subset, subtitle):
                err_subset = np.asarray(err_subset)
                cat_subset = np.asarray(cat_subset)

                mask = np.isfinite(err_subset) & np.isfinite(cat_subset)
                err_subset = err_subset[mask]
                cat_subset = cat_subset[mask]

                if err_subset.size == 0:
                    ax.set_title(subtitle + "\n(empty)")
                    ax.grid(True, linestyle="--", alpha=0.4)
                    return

                bias, std, rmse, mae = _compute_stats(err_subset)

                # --- normalization with median of category ---
                med_cat = np.median(cat_subset)
                norm_bias = bias / med_cat if med_cat != 0 else np.nan

                ax.hist(err_subset, bins=40)
                ax.set_title(subtitle)
                ax.set_xlabel(f"Error ({unit})")
                ax.set_ylabel("Count")
                ax.grid(True, linestyle="--", alpha=0.4)

                txt = (f"bias = {bias:.2f} {unit}\n"
                    f"norm_bias = {norm_bias:.4f} (bias/median)\n"
                    f"median(cat) = {med_cat:.2f}\n"
                    f"stddev = {std:.2f} {unit}\n"
                    f"rmse = {rmse:.2f} {unit}\n"
                    f"mae = {mae:.2f} {unit}\n"
                    f"n = {err_subset.size}")

                ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                        ha="right", va="top", fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

                if xlim is not None:
                    ax.set_xlim(xlim)

            # Panel 1: all
            draw(axes[0], errors, cat_values, f"{title_prefix}\nAll cases")

            # Panels 2–4: categories
            for k in range(3):
                lo, hi = bins[k], bins[k + 1]

                if k < 2:
                    sel = (cat_values >= lo) & (cat_values < hi)
                else:
                    sel = (cat_values >= lo) & (cat_values <= hi)

                subtitle = f"{title_prefix}\nCat{k+1}: [{lo:.1f}, {hi:.1f}]"
                draw(axes[k + 1], errors[sel], cat_values[sel], subtitle)

            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
        
        


        # ---------------------------------------------------------------------------------------------------------------------------------------------------
        # On ne visualise que le canal IRWIN (canal 0)
        ch = 4 if x.shape[1] >4 else 0
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
        
        if self.log_wind:
            sar_denorm = np.exp(sar_denorm) - 1e-10
            pred_denorm = np.exp(pred_denorm) - 1e-10

        if self.output_data == "aam":
            sar_denorm = moment_to_sar(sar_denorm)
            pred_denorm = moment_to_sar(pred_denorm)

        # Conversion numpy

        B,H,W = sar_denorm.shape
        if self.crop_sar:
            ir_denorm = ir_denorm[:,W//2-W//4:W//2+W//4,H//2-H//4:H//2+H//4]
            sar_denorm = sar_denorm[:,W//2-W//4:W//2+W//4,H//2-H//4:H//2+H//4]
            pred_denorm = pred_denorm[:,W//2-W//4:W//2+W//4,H//2-H//4:H//2+H//4]

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
        
        if self.crop_sar : 
            mask_np = mask_np[:,W//2-W//4:W//2+W//4,H//2-H//4:H//2+H//4]
        B, H, W = pred_np.shape

        pred_vmax = np.full(B, np.nan, dtype=np.float32)
        pred_rmax_km    = np.full(B, np.nan, dtype=np.float32)

        for i in range(B):
            vmax1d_pred, rmax1d_pred_km = self.compute_vmax1d_rmax1d(pred_np[i])
            pred_vmax[i] = vmax1d_pred
            pred_rmax_km[i] = rmax1d_pred_km # 2 km per pixel

        # ==========================
        # Analysis values in same units
        # ==========================
        analysis_vmax = np.array(analysis_vmax, dtype=np.float32)  # m/s 
        analysis_rmax = np.array(analysis_rmax, dtype=np.float32)  # meters
        analysis_rmax_km = analysis_rmax / 1000.0 # to km

        # filter missing analysis
        ok_vmax = np.isfinite(analysis_vmax) & np.isfinite(pred_vmax)
        ok_rmax = np.isfinite(analysis_rmax_km) & np.isfinite(pred_rmax_km)

        # Errors
        err_vmax = (pred_vmax - analysis_vmax)[ok_vmax]
        cat_vmax = analysis_vmax[ok_vmax]  # categories based on analysis on m/*s
        err_rmax = (pred_rmax_km - analysis_rmax_km)[ok_rmax]
        cat_rmax = analysis_rmax_km[ok_rmax]     # categories based on analysis

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
                bins = [0,32,49,80] ,  # m/s,
                title_prefix="Vmax error",
                unit="m/s",
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
        sar_knots_flat  = sar_valid
        pred_knots_flat = pred_valid

        # → Seulement pour la distribution
        distdata.compare_sar_distribution(
            sar_knots_flat,
            pred_knots_flat,
            self.output_dir,
            set=set,
            epoch=epoch
        )

        sar_2d  = sar_denorm   # (B, H, W)
        pred_2d = pred_denorm # (B, H, W)
        mask_2d = mask_np 
        os.makedirs(os.path.join(self.output_dir,'predictions_denormalisees',set),exist_ok=True)                               # (B, H, W)
        with open(os.path.join(self.output_dir,"predictions_denormalisees",set,"predictions_denormalisées.pkl"),"wb") as f:
            pkl.dump({f"{set}":[ir_np,sar_2d,pred_2d,mask_2d,infos_np],"model":model},f)

        # Conversion en knots
        sar_2d  = sar_2d #m/s
        pred_2d= pred_2d #m/s


        # → Vmax utilise les champs 2D
        for min,max in zip([19,63,83,96,113],[63,83,96,113,200]):
            distdata.vmax_compare(
                analysis_vmax,
                pred_2d*mask_2d,
                self.output_dir,
                set=set,
                epoch=epoch,
                min=min,
                max=max
            )
        distdata.vmax_compare(
                analysis_vmax,
                pred_2d*mask_2d,
                self.output_dir,
                set=set,
                epoch=epoch
            )
        

        for min,max in zip([0,30,60],[30,60,100]):
            distdata.rmax_compare(
                analysis_rmax,
                pred_2d*mask_2d,
                self.output_dir,
                set=set,
                epoch=epoch,
                min=min,
                max=max
            )
        distdata.rmax_compare(
                analysis_rmax,
                pred_2d*mask_2d,
                self.output_dir,
                set=set,
                epoch=epoch
            )

        # → Radial Vmax utilise aussi les champs 2D
        distdata.compare_radial_vmax(
            sar_2d,
            pred_2d*mask_2d,
            output_dir=self.output_dir,
            set=set,
            epoch=epoch,
            plot=False
        )

        # mae 
        distdata.compute_mae_metric(
            sar_2d,
            pred_2d*mask_2d,
            output_dir=self.output_dir,
            set=set,
            epoch=epoch,
            plot=False
        )
       
        for i in sample_ids:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            H, W = sar_np[i].shape
            cx, cy = W // 2, H // 2

            

            
            rmax = analysis_rmax[i] / 1000    # en km
            vmax = analysis_vmax[i]   # m/s 
            if vmax is None or np.isnan(rmax):
                vmax = 99999
            if rmax is None or np.isnan(rmax):
                rmax = 99999

            # IRWIN
            distdata.plot_ir(ir_np[i], cmap=self.cmap_ir,ax=axes[0],fig=fig,x_lim=H)
            axes[0].set_title("IRWIN (°C)")
            axes[0].axis("off")

            # === TRUE SAR ===
            sar_vis = np.where(mask_np[i]==1, sar_np[i], np.nan)
            # Compute radial metrics
            vmax1d_pred, rmax1d_pred = self.compute_vmax1d_rmax1d(pred_np[i])
            vmax_sar, rmax_sar = self.compute_vmax1d_rmax1d(sar_np[i])
            distdata.plot_sar(sar_vis, cmap=self.cmap_sar,ax=axes[1],fig=fig,x_lim=H)
            axes[1].set_title("True SAR (m/s)")
            axes[1].axhline(y=0, color="black", linewidth=1)
            axes[1].axvline(x=0, color="black", linewidth=1)

            if rmax is not None:
                axes[1].add_patch(Circle((0, 0), radius=rmax, color="black", fill=False, linestyle="--"))

            axes[1].axis("off")

            # === PREDICTED SAR ===
            # pred_vis = np.where(mask_np[i]==1, pred_np[i], np.nan)
            distdata.plot_sar(pred_np[i], cmap=self.cmap_sar,ax=axes[2],fig=fig,x_lim=H)
            axes[2].set_title("Predicted SAR (m/s)")
            axes[2].axhline(y=0, color="black", linewidth=1)
            axes[2].axvline(x=0, color="black", linewidth=1)

            if rmax is not None:
                axes[2].add_patch(Circle((0, 0), radius=rmax, color="black", fill=False, linestyle="--"))

            axes[2].axis("off")

            # === TITLE including Rmax1D and Vmax1D ===
            fig.suptitle(
                f"Cyclone: {cyclone_id[i]} — SAR Time: {sar_time[i]} — Epoch {epoch+1}\n"
                f"Analysis Rmax = {rmax:.1f} Km — Rmax SAR = {rmax_sar:.1f} km — Predicted Rmax1D = {rmax1d_pred:.1f} Km\n"
                f"Anaysis Vmax = {vmax:.1f} m/s — Vmax SAR = {vmax_sar:.1f} m/s — Predicted Vmax1D = {vmax1d_pred:.1f} m/s",
                fontsize=10
            )


            plt.tight_layout()
            save_path = os.path.join(self.output_dir, "samples",set,  f"sample_{i}_epoch_{epoch+1}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

        print(f"💾 Saved {num} sample images.")
        set_compare = "train" if self.cfg.code_test else "test"
        if set == set_compare and self.cfg.use_flow_matching:
            os.makedirs(os.path.join(self.output_dir,f"fm_diagnostics_{set}","rank_hist_and_samples"), exist_ok=True)
            rank_smple_path = os.path.join(self.output_dir,f"fm_diagnostics_{set}","rank_hist_and_samples")
            
            print(f"starting fm diagnostics for {set} set")
            #loop over thye batch with tqdm to see the progress
            mean_ph, std_ph = [], []
            check_indices = [len(x_fm) // k for k in [8, 7, 6, 5, 4, 3, 2]]
            for i, (x_ir, sar_target, mask, infos) in enumerate(zip(x_fm, sar_fm, mask_fm, infos_fm)):
                x_ir = x_ir.unsqueeze(0).to(device)
                sar_target = sar_target.unsqueeze(0).to(device)
                mask = mask.unsqueeze(0).to(device)
                fm_model = model
                stats = {"mean": self.mean_sar, "std": self.std_sar}
                if not self.cfg.use_residu : 
                    ensemble = fm_inf.generate_ensemble(model=fm_model, ir_input=x_ir, n_members=20, device=device)
                else : 
                    mean_pred = reg_model(x_ir,timestep=0).sample
                    ensemble = fm_inf.generate_residual_ensemble(fm_model,mean_pred,
                                                                 residual_mean=resid_stats["mean"],
                                                                 residual_std=resid_stats["std"])
                    
                

                if i in check_indices:
                    os.makedirs(os.path.join(rank_smple_path, f"{cyclone_id[i]}_{sar_time[i]}"), exist_ok=True)
                    
                    save_path_samples = os.path.join(rank_smple_path, f"{cyclone_id[i]}_{sar_time[i]}", "samples.png")
                    save_path_rank =    os.path.join(rank_smple_path, f"{cyclone_id[i]}_{sar_time[i]}", "rank_histogram.png")
                    ens_mean_phys, ens_std_phys = fm_inf.plot_ensemble_results(x_ir,sar_target=sar_target, ensemble=ensemble, stats=stats, mask=mask, save_path= save_path_samples,
                                                cmap_sar=self.cmap_sar, cmap_ir=self.cmap_ir,save_pic=True)
                    mean_ph.append(ens_mean_phys); std_ph.append(ens_std_phys)

                    #rank histogram of the analysis in the ensemble
                    ranks , n_members= fm_inf.rank_histogram(ensemble, sar_target.unsqueeze(0), mask)
                    fm_inf.plot_rank_histogram(ranks, n_members=n_members, save_pth=save_path_rank)
                else : 
                    ens_mean_phys, ens_std_phys = fm_inf.plot_ensemble_results(x_ir,sar_target=sar_target, ensemble=ensemble, 
                                                                               stats=stats, mask=mask, save_path= None,
                                                cmap_sar=self.cmap_sar, cmap_ir=self.cmap_ir,save_pic=False)
                    mean_ph.append(ens_mean_phys); std_ph.append(ens_std_phys)




            #plot_vmax plots and rmax plot with mean_ph vs sar and std_ph vs sar for all the samples in the batch
            mean_ph = np.array(mean_ph)
            std_ph = np.array(std_ph)
            sar_target = sar_target.squeeze().cpu().numpy()
            os.makedirs(os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots"), exist_ok=True)
            save_path = os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots", f"mean_std_vmax_rmax_epoch_{epoch+1}.png")
            distdata.vmax_compare(
                analysis_vmax,
                mean_ph*mask_2d,
                self.output_dir,
                set=set,
                epoch=epoch,
                y_label= "Vmax Mean ensemble fm (m/s)",
                output = os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots", f"mean_vmax.png")
            )
            distdata.rmax_compare(
                analysis_rmax,
                mean_ph*mask_2d,
                self.output_dir,
                set=set,
                epoch=epoch,
                y_label= "Rmax Mean ensemble fm (km)",
                output= os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots", f"mean_rmax_epoch.png")
            )
            # Stabdard deviation plots
            distdata.vmax_compare(
                analysis_vmax,
                std_ph*mask_2d,
                self.output_dir,
                set=set,
                epoch=epoch,
                y_label= "Vmax Std ensemble fm (m/s)",
                output = os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots", f"std_vmax.png")
            )
            distdata.rmax_compare(
                analysis_rmax,
                std_ph*mask_2d,
                self.output_dir,
                set=set,
                epoch=epoch,
                y_label= "Rmax Std ensemble fm (km)",
                output= os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots", f"std_rmax.png")
            )

            distdata.compare_radial_vmax(
                sar_2d,
                mean_ph*mask_2d,
                output_dir=self.output_dir,
                set=set,
                epoch=epoch,
                output = os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots", f"Mean_radial_vmax.png"),
                plot=False
            )

            # mae 
            distdata.compute_mae_metric(
                sar_2d,
                mean_ph*mask_2d,
                output_dir=self.output_dir,
                set=set,
                epoch=epoch,
                output = os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots", f"Mean_mae.png"),
                plot=False
            )

            distdata.compare_radial_vmax(
                sar_2d,
                std_ph*mask_2d,
                output_dir=self.output_dir,
                set=set,
                epoch=epoch,
                output = os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots", f"Std_radial_vmax.png"),
                plot=False
            )

            # mae 
            distdata.compute_mae_metric(
                sar_2d,
                std_ph*mask_2d,
                output_dir=self.output_dir,
                set=set,
                epoch=epoch,
                output = os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots", f"Std_mae.png"),
                plot=False
            )
            # create flatten data
            sar_flat = sar_2d.flatten()
            mean_ph_flat = mean_ph.flatten()
            std_ph_flat = std_ph.flatten()
            mask_flat = mask_2d.flatten()
            valid = mask_flat == 1
            sar_valid = sar_flat[valid]
            mean_ph_valid = mean_ph_flat[valid]
            std_ph_valid = std_ph_flat[valid]
            distdata.compare_sar_distribution(sar_valid, mean_ph_valid, self.output_dir, set, epoch,
                            output=os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots", f"mean_distribution.png"))
            distdata.compare_sar_distribution(sar_valid, std_ph_valid, self.output_dir, set, epoch,
                            output=os.path.join(self.output_dir,f"fm_diagnostics_{set}","mean_std_plots", f"std_distribution.png"))


            



        
    
    def plot_fm_diagnostics(self, model, x_ir, sar_target, mask, stats, device="cpu",
                        num_steps=20, n_rows=5, set="validation", epoch=0, cmap_sar="viridis", cmap_ir="gray",
                        reg_model=None,resid_stats=None):
        """
        Quick validation figure: IR | SAR target | FM sample | velocity at t=0.5.
        Call this from LogValidationSamples.on_validation_plots() with use_flow_matching=True.
        """
        model.eval()
        B = min(x_ir.shape[0], n_rows)
        x_ir    = x_ir[:B].to(device)
        sar_tgt = sar_target[:B].to(device)
        mask_v  = mask[:B].to(device)

        if sar_tgt.ndim == 3:
            sar_tgt = sar_tgt.unsqueeze(1)
            mask_v  = mask_v.unsqueeze(1)

        with torch.no_grad():
            # One FM sample
            if not self.cfg.use_residu:
                z      = torch.randn(B, 1, x_ir.shape[2], x_ir.shape[3], device=device)
                fm_out = _euler_ode(model, z, x_ir, num_steps, device)
                # Velocity at midpoint t=0.5 (diagnostic: should be near x₁−z)
                t_mid = torch.full((B,), 0.5, device=device)
                mi    = torch.cat([z, x_ir], dim=1)
                vel05 = model(mi, t_mid).sample

            else : 
                B,_,H,W = x_ir.shape
                z      = torch.randn(B, 1, x_ir.shape[2], x_ir.shape[3], device=device)
                mean_pred = reg_model(x_ir, timestep=0).sample

                resid_norm = fm_inf.ode_solver_residual(
                    fm_model=model,
                    z=z,
                    mean_pred=mean_pred,
                    num_steps=self.cfg.fm_num_inference_steps
                )
                fm_out = fm_inf.reconstruct_from_residual(
                    resid_norm,
                    mean_pred,
                    resid_stats["mean"],
                    resid_stats["std"]
                )



        sar_mean = stats["mean"]
        sar_std  = stats["std"]

        fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
        if B == 1:
            axes = axes[None]
        #create a folder for fm diagnostics
        
        for i in range(B):
            im_ir  = x_ir[i, 0].cpu().numpy()          # first IR channel
            im_tgt = sar_tgt[i, 0].cpu().numpy() * sar_std + sar_mean
            im_fm  = fm_out[i, 0].cpu().numpy()  * sar_std + sar_mean
            im_vel = vel05[i, 0].cpu().numpy() if not self.cfg.use_residu else (resid_norm[i,0]*sar_std + sar_mean).cpu().numpy()

            
            distdata.plot_ir(im_ir, fig=fig, ax=axes[i,0], cmap=cmap_ir);         axes[i, 0].set_title("IR (ch 0)")
            distdata.plot_sar(im_tgt, fig=fig, ax=axes[i,1], cmap=cmap_sar or "RdBu_r");       axes[i, 1].set_title("SAR target")
            distdata.plot_sar(im_fm, fig=fig, ax=axes[i,2], cmap=cmap_sar or "RdBu_r");       axes[i, 2].set_title("FM sample")
            distdata.plot_sar(im_vel, fig=fig, ax=axes[i,3], cmap= "seismic");       axes[i, 3].set_title("Velocity @ t=0.5" if not self.cfg.use_residu else "Residu")
       
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.join(self.output_dir,f"fm_diagnostics_{set}"),exist_ok=True)
        save_path = os.path.join(self.output_dir, f"fm_diagnostics_{set}", f"fm_diagnostics.png")
        plt.savefig(save_path, dpi=150)
        
    
        if self.cfg.use_residu : 
            for i in range(B) :
                im_ir  = x_ir[i, 0].cpu().numpy()          # first IR channel
                im_tgt = sar_tgt[i, 0].cpu().numpy() * sar_std + sar_mean
                im_fm  = fm_out[i, 0].cpu().numpy()  * sar_std + sar_mean

                reg_pred = reg_model(x_ir[i,:,:,:].unsqueeze(0),timestep=0).sample
                residual_ensemble = fm_inf.generate_residual_ensemble(model,reg_pred,20,
                                                                    self.cfg.fm_num_inference_steps,
                                                                    resid_stats["mean"],
                                                                    resid_stats["std"]).cpu().numpy()

                fig2 = distdata.plot_comparison(im_ir,im_tgt,mask_v[i,0].cpu().numpy(),stats,reg_pred.cpu().numpy(),residual_ensemble)
        
                plt.savefig(os.path.join(self.output_dir,f"fm_diagnostics_{set}",f"plot_comparison_residual_sample_{i}.png"))
                plt.close(fig2)

    

    def on_validation_plots(self, model, epoch, dataloader, device, reg_model=None, resid_stats=None):
            print(f"📸 Logging validation samples at epoch {epoch +1}")

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
                    # inf = list of dicts, one per sample
                keep = [d.get("augmentation", 0) == 0 for d in inf]

                if not any(keep):
                    continue

                keep_idx = torch.tensor(keep, device=ir.device)

                all_ir_train.append(ir[keep_idx])
                all_sar_train.append(sar[keep_idx])
                mask_train.append(mask[keep_idx])
                infos_train.append([d for d, k in zip(inf, keep) if k])
            
            for ir, sar, mask, inf in dataloader[2]:
                all_ir_test.append(ir)
                all_sar_test.append(sar)
                mask_test.append(mask)
                infos_test.append(inf)
            
            #anggrek data
            

        

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


            # Créer un tuple exactement comme un batch
            batch_full_val = (ir_full_val, sar_full_val, mask_val, infos_val)
            batch_full_train = (ir_full_train, sar_full_train, mask_train, infos_train)
            batch_full_test = (ir_full_test, sar_full_test, mask_test, infos_test)
            # self.log_batch(model, batch_full_train, epoch, device, set="train",reg_model=reg_model,resid_stats=resid_stats)
            if not self.cfg.code_test : 
                self.log_batch(model, batch_full_val, epoch, device,reg_model=reg_model,resid_stats=resid_stats)
                self.log_batch(model, batch_full_test, epoch, device, set="test",reg_model=reg_model,resid_stats=resid_stats)
             
            if not  self.conditional_model: 

                for ir, sar, mask, inf in dataloader[3]:
                    ir_anggrek.append(ir)
                    sar_anggrek.append(sar)
                    mask_anggrek.append(mask)
                    infos_anggrek.append(inf)
                
                ir_full_anggrek = torch.cat(ir_anggrek, dim=0)
                sar_full_anggrek = torch.cat(sar_anggrek, dim=0)
                mask_anggrek = torch.cat(mask_anggrek, dim=0)
                infos_anggrek = [d for batch in infos_anggrek for d in batch]
                batch_full_anggrek = (ir_full_anggrek, sar_full_anggrek, mask_anggrek, infos_anggrek)
                

                self.anggrek_plots(model, batch_full_anggrek, epoch, device,reg_model=reg_model, resid_stats= resid_stats)
                print("finished anggrek plots")
            
            if self.cfg.use_flow_matching:
                print("starting fm diagnostics for train, val and test sets")
                if self.cfg.code_test : 
                    self.plot_fm_diagnostics(model, ir_full_train, sar_full_train, mask_train, stats={"mean": self.mean_sar, "std": self.std_sar}, 
                                            device=device, num_steps=self.cfg.fm_num_inference_steps, n_rows=5, set="train", epoch=epoch,
                                            cmap_sar=self.cmap_sar, cmap_ir=self.cmap_ir,reg_model=reg_model,resid_stats=resid_stats)
                if not self.cfg.code_test :
                    
                    self.plot_fm_diagnostics(model, ir_full_val, sar_full_val, mask_val, stats={"mean": self.mean_sar, "std": self.std_sar}, 
                                            device=device, num_steps=self.cfg.fm_num_inference_steps, n_rows=5, set="validation", epoch=epoch,
                                            cmap_sar=self.cmap_sar, cmap_ir=self.cmap_ir,reg_model=reg_model,resid_stats=resid_stats)
                    self.plot_fm_diagnostics(model, ir_full_test, sar_full_test, mask_test, stats={"mean": self.mean_sar, "std": self.std_sar}, 
                                            device=device, num_steps=self.cfg.fm_num_inference_steps, n_rows=8, set="test", epoch=epoch, 
                                            cmap_sar=self.cmap_sar, cmap_ir=self.cmap_ir,reg_model=reg_model,resid_stats=resid_stats)  

                self.anggrek_plots(model, batch_full_anggrek, epoch, device, add_mean_std_fm = True, reg_model=reg_model,resid_stats=resid_stats)
            
                #ode selver with guidance
                if not self.cfg.use_residu : 
                    ode_guidances = [];ode_reprojections = []
                    for ir, sar, mask, inf in dataloader[0 if self.cfg.code_test else 2]:   #test set
                        ir = ir.to(device)
                        sar = sar.to(device)
                        mask = mask.to(device)
                        fm_model = model
                        stats = {"mean": self.mean_sar, "std": self.std_sar}
                        z = torch.randn_like(sar)
                        ode_guidance_out = fm_inf.ode_solver_with_guidance(model=fm_model, z=z, ir_input=ir, obs_sar=sar, obs_mask=mask, 
                                                                        num_steps=self.cfg.fm_num_inference_steps,
                                                                            sar_mean_stat=self.mean_sar, sar_std_stat=self.std_sar)
                        ode_reprojectiion_out = fm_inf.ode_solver_with_reprojection(model=fm_model,z=z, ir_input=ir, obs_sar=sar, 
                                                                                    obs_mask=mask, num_steps=self.cfg.fm_num_inference_steps)
                        
                        ode_guidances.append(ode_guidance_out)
                        ode_reprojections.append(ode_reprojectiion_out)

                    ode_guidances = torch.cat(ode_guidances, dim=0)
                    ode_reprojections = torch.cat(ode_reprojections, dim=0)
                    batch_full_test_guidance = (ir_full_test,  sar_full_test, mask_test, infos_test, ode_guidances) if not self.cfg.code_test else (ir_full_train,  sar_full_train, mask_train, infos_train, ode_guidances)     #
                    batch_full_test_reprojection = (ir_full_test,  sar_full_test, mask_test, infos_test, ode_reprojections) if not self.cfg.code_test else (ir_full_train,  sar_full_train, mask_train, infos_train, ode_reprojections) #
                    self.log_batch(model, batch_full_test_guidance, epoch, device, set="test_guidance",ode_pred=True,reg_model=reg_model,resid_stats=resid_stats)
                    self.log_batch(model, batch_full_test_reprojection, epoch, device, set="test_reprojection", ode_pred=True,reg_model=reg_model,resid_stats=resid_stats)

        # save distribution of wind speed of pred and true val to compare it
        #just in the lkast epoch

    

    def anggrek_plots(self, model, batch, epoch, device, add_mean_std_fm = False,reg_model=None,resid_stats=None):
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


        x, _, _, infos = batch
        x = x.to(device)

        # Predict
        with torch.no_grad():
            if not add_mean_std_fm:
                if not self.conditional_model:
                    if not self.cfg.use_flow_matching:
                        print(x.shape)
                        pred = model(x, timestep=0).sample
                    elif not self.cfg.use_residu:
                        B = x.shape[0]
                        H = x.shape[2]
                        W = x.shape[3]
                        z = torch.randn(B, 1, H, W, device=device)
                        pred = _euler_ode(model, z, x, self.cfg.fm_num_inference_steps, device)
                    else : 
                        B, _, H, W = x.shape
                        z = torch.randn(B, 1, H, W, device=device)
                        mean_pred = reg_model(x, timestep=0).sample

                        resid_norm = fm_inf.ode_solver_residual(
                            fm_model=model,
                            z=z,
                            mean_pred=mean_pred,
                            num_steps=self.cfg.fm_num_inference_steps
                        )

                        pred = fm_inf.reconstruct_from_residual(
                            resid_norm,
                            mean_pred,
                            resid_stats["mean"],
                            resid_stats["std"]
                        )

                        

                else :
                    shear = torch.stack([
                                            torch.as_tensor(d["shear"], dtype=torch.float32)
                                            for d in infos
                                        ]).to(device) 
                    pred = model(x, timestep=0, cond=shear).sample

        # -------------------------
        def denorm(t, mean, std):
            return t * (std + 1e-10) + mean

        def annular_denormalization(images_norm, stats, bin_size=1):
            # images_norm: (B,H,W)
            N, H, W = images_norm.shape
            cx, cy = W// 2, H // 2
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

        # -------------------------
        if not add_mean_std_fm:
            x_np = x.detach().cpu().numpy()  # (B,C,H,W)
            pred_np = pred.detach().squeeze().cpu().numpy()  # (B,H,W) or (H,W) if B=1

            if pred_np.ndim == 2:
                pred_np = pred_np[None, ...]  # force (B,H,W)

            B = x_np.shape[0]
            
            ch = 2 if x_np.shape[1] > 4 else 0   # IR channel index
            
            ir = x_np[:, ch, :, :]  # (B,H,W)
            if self.add_era5 : 
                era5 = x_np[:,-1,:,:]
                era5 = denorm(era5,self.mean_X[-1],self.std_X[-1])
            pred1 = pred_np          # (B,H,W)

            ir_den = denorm(ir, self.mean_X[ch], self.std_X[ch])

            if self.norm == "z_score":
                pred_den = denorm(pred1, self.mean_sar, self.std_sar)
            elif self.norm == "annular":
                pred_den = annular_denormalization(pred1, stats={"mean": self.mean_sar, "std": self.std_sar})
            else:
                # fallback: assume already denorm
                pred_den = pred1
            
            if self.log_wind:
                pred_den = np.exp(pred_den) - 1e-10

            if self.output_data == "aam":
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

            if self.crop_sar:
                ir_den = ir_den[:,W//2-W//4:W//2+W//4,H//2-H//4:H//2+H//4]
                pred_den = pred_den[:,W//2-W//4:W//2+W//4,H//2-H//4:H//2+H//4]
                era5 = era5[:,W//2-W//4:W//2+W//4,H//2-H//4:H//2+H//4]
        
            os.makedirs(os.path.join(self.output_dir,'predictions_denormalisees',"anggrek"),exist_ok=True)                               # (B, H, W)
            with open(os.path.join(self.output_dir,"predictions_denormalisees","anggrek","predictions_denormalisées.pkl"),"wb") as f:
                pkl.dump({f"anggrek":[ir_den,pred_den,infos_ord],"model":model},f)


            out_root = Path(self.output_dir) / "anggrek_monitoring"
            field_dir = out_root / "field_plots"
            field_dir.mkdir(parents=True, exist_ok=True)
        
        else : 
            os.makedirs(os.path.join(self.output_dir,"fm_diagnostics_anggrek","anggrek_monitoring"),exist_ok=True) 
            out_root = Path(self.output_dir) / "fm_diagnostics_anggrek" / "anggrek_monitoring"
            os.makedirs(os.path.join(self.output_dir,"fm_diagnostics_anggrek","prediction_denormalisées"),exist_ok=True)
            os.makedirs(os.path.join(self.output_dir,"fm_diagnostics_anggrek","anggrek_monitoring","fields_plots"),exist_ok=True)
            

            #x contient les champs d'entrée du modèle, dont le dernier est ERA5 si add_era5=True
            mean_ph, std_ph = [], []
            sar_time = [d.get("date") for d in infos]
            time_parsed = pd.to_datetime(sar_time, errors="coerce")
            order = np.argsort(time_parsed.values.astype("datetime64[ns]"))
            time_parsed = time_parsed[order]
            infos_ord = [infos[i] for i in order]
            for i,x_fm in enumerate(x):
                if not self.cfg.use_residu:
                    ensemble = fm_inf.generate_ensemble(model=model, ir_input=x_fm,device=device)
                else : 
                    if x_fm.ndim == 3 :
                        x_fm = x_fm.unsqueeze(0)
                    mean_pred = reg_model(x_fm,timestep=0).sample
                    ensemble = fm_inf.generate_residual_ensemble(model,mean_pred,num_steps=self.cfg.fm_num_inference_steps,residual_mean=resid_stats["mean"],
                                                                 residual_std=resid_stats["std"])
                    
                mean_fm, std_fm = fm_inf.plot_ensemble_results(x_fm, sar_target=None, ensemble=ensemble,
                                                        stats={"mean": self.mean_sar, "std": self.std_sar}, mask=None, 
                                                    save_path=os.path.join(self.output_dir,"fm_diagnostics_anggrek","anggrek_monitoring","fields_plots",f"{time_parsed[i].strftime('%Y%m%d%H%M%S')}_ensemble.png"),
                                                    cmap_sar=self.cmap_sar, cmap_ir=self.cmap_ir, save_pic=True,
                                                            )
                mean_ph.append(mean_fm); std_ph.append(std_fm)
            mean_ph = np.array(mean_ph); std_ph = np.array(std_ph)
            x_np = x.detach().cpu().numpy()
            #save as a pkl 
            with open(os.path.join(self.output_dir,"fm_diagnostics_anggrek","prediction_denormalisées","mean_std_fm.pkl"),"wb") as f:
                pkl.dump({"mean_fm": mean_ph, "std_fm": std_ph, "time": time_parsed,"ir_norm": x_np,"model": model},f)





        vmax = np.array([d.get("vmax", np.nan) for d in infos_ord], dtype=float)


        pixel_km = 2.0

        if not add_mean_std_fm:
            for i in range(B):
                t = time_parsed[i]
                H, W = pred_den[i].shape
                cx, cy = W // 2, H // 2

                vmax_pred, rmax_pred = self.compute_vmax1d_rmax1d(pred_den[i])
                

                if pd.isna(t):
                    date_key = f"unknown_{i:03d}"
                    fname = f"unknown_{i:03d}_fields.png"
                    supt = f"Unknown time (idx={i})"
                else:
                    date_key = t.strftime("%Y%m%d%H%M%S")  
                    fname = f"{date_key}_fields.png"
                    supt = (
                                f"{t.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                f"Best Track Vmax: {vmax[i]:.2f} m/s | "
                                f"Pred Vmax: {vmax_pred:.2f} m/s | "
                                f"Pred RMW: {rmax_pred:.1f} km"
                            )
                sub = field_dir
                sub.mkdir(parents=True, exist_ok=True)

                fig, axs = plt.subplots(1, 3 if self.add_era5 else 2, figsize=(8, 4), constrained_layout=True)

                distdata.plot_ir(ir_den[i], cmap=self.cmap_ir,ax=axs[0],fig=fig,x_lim=H)
                axs[0].set_title("IRWIN")
                axs[0].axis("off")
                distdata.plot_sar(pred_den[i], cmap=self.cmap_sar,ax=axs[2 if self.add_era5 else 1],fig=fig, x_lim=H)
                axs[2 if self.add_era5 else 1].set_title("Prediction")
                axs[2 if self.add_era5 else 1].axhline(y=0, color="black", linewidth=1)
                axs[2 if self.add_era5 else 1].axvline(x=0, color="black", linewidth=1)
                if self.add_era5:
                    distdata.plot_sar(annular_denormalization(era5[i], stats={"mean": self.mean_sar, "std": self.std_sar}) if self.norm == "annular" else era5[i]*self.std_sar+self.mean_sar
                     , cmap=self.cmap_sar,ax=axs[1],fig=fig,x_lim=H)
                    axs[1].set_title("ERA 5")
                    axs[1].axis("off")

                # cercle RMW
                axs[2 if self.add_era5 else 1].add_patch(
                                Circle((0, 0), radius=rmax_pred, color="black", fill=False, linestyle="--")
                                )
                fig.suptitle(supt)
                fig.savefig(sub / fname, dpi=150)
                plt.close(fig)

        #Resample SAr from 2km to 3km
        @torch.no_grad()
        def resample_2km_to_3km_torch(xs):
            resamples = []

            for x in xs:
                
                x = torch.as_tensor(x, dtype=torch.float32, device="cpu")

                # HxW -> 1x1xHxW
                if x.dim() == 2:
                    x = x.unsqueeze(0).unsqueeze(0)

                H, W = x.shape[-2:]
                in_res = 4 if self.regrid_ir else 2
                new_H = int(H * in_res / 3) 
                new_W = int(W * in_res / 3)

                # CPU interpolation
                x3 = F.interpolate(x, size=(new_H, new_W), mode="bilinear", align_corners=False)

                # back to numpy (HxW)
                resamples.append(x3.squeeze(0).squeeze(0).cpu().numpy())

            return np.stack(resamples, axis=0)
        if not add_mean_std_fm:
            pred_denorm_3m = resample_2km_to_3km_torch(pred_den)
            pred_vmax = np.nanmax(pred_denorm_3m.reshape(len(pred_den), -1), axis=1)
            pred_vmax_2km = np.nanmax(pred_den.reshape(len(pred_den), -1), axis=1)
        else : 
            mean_fm_3m = resample_2km_to_3km_torch(mean_ph); std_fm_3m = resample_2km_to_3km_torch(std_ph)
            pred_vmax_mean_fm_3m = np.nanmax(mean_fm_3m.reshape(len(mean_ph), -1), axis=1)
            pred_vmax_std_fm_3m = np.nanmax(std_fm_3m.reshape(len(std_ph), -1), axis=1)



        vmax = np.array([d.get("vmax", np.nan) for d in infos_ord], dtype=float)
        vmax_cyclobs = np.array([d.get("vmax_cyclobs", np.nan) for d in infos_ord], dtype=float)
        analysis_vmax_cyclobs = np.array([d.get("analysis_vmax_cyclobs", np.nan) for d in infos_ord], dtype=float)
        ibtracs_vmax = np.array([d.get("ibtracs_vmax", np.nan) for d in infos_ord], dtype=float)
        satcon_vmax  = np.array([d.get("satcon_vmax", np.nan) for d in infos_ord], dtype=float)
        era5_vmaxs = np.array([d.get("era5_vmax", np.nan) for d in infos_ord], dtype=float)

        
        if not add_mean_std_fm:
            ok = np.isfinite(vmax) & np.isfinite(pred_vmax)
            rmse = float(np.sqrt(np.mean((pred_vmax[ok] - vmax[ok]) ** 2))) if np.any(ok) else np.nan
        else:
            ok = np.isfinite(vmax) & np.isfinite(pred_vmax_mean_fm_3m)
            rmse = float(np.sqrt(np.mean((pred_vmax_mean_fm_3m[ok] - vmax[ok]) ** 2))) if np.any(ok) else np.nan

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(time_parsed, ibtracs_vmax, color="black", linewidth=2, label="IBTrACS (Best Track)")
        ax.plot(time_parsed, vmax, color="red", linewidth=2, label="ATCF")
        ax.plot(time_parsed, satcon_vmax, color="blue", linewidth=2, label="SATCON")
        ax.plot(time_parsed, era5_vmaxs, color="orange", linewidth=2, label="ERA5")

        if not add_mean_std_fm:
            ax.plot(time_parsed, pred_vmax, color="green", linewidth=2, label="UNET Res 3 km")
            ax.plot(time_parsed, pred_vmax_2km, color="magenta", linewidth=2, label="UNET Res 2 km")
        else:
            ax.plot(time_parsed, pred_vmax_mean_fm_3m, color="green", linewidth=2, label="flow matching (mean n=20) Res 3 km")
            ax.fill_between(
                    time_parsed,
                    pred_vmax_mean_fm_3m - pred_vmax_std_fm_3m,
                    pred_vmax_mean_fm_3m + pred_vmax_std_fm_3m,
                    color="magenta",
                    alpha=0.10,
                    label="Residual FM ± std"
                )
            

        analysis_arr = np.array(analysis_vmax_cyclobs, dtype=float)
        cyclobs_arr  = np.array(vmax_cyclobs, dtype=float)

        t = np.array(time_parsed)
        analysis_arr = np.array(analysis_vmax_cyclobs, dtype=float)
        cyclobs_arr  = np.array(vmax_cyclobs, dtype=float)

        t = np.array(time_parsed)

        mA = ~np.isnan(analysis_arr)
        mC = ~np.isnan(cyclobs_arr)


        # --- Markers (high-quality style)
        ax.scatter(
            t[mA], analysis_arr[mA],
            marker="*", s=110,                  # size
            facecolor="#FFD54F",                # warm yellow
            edgecolor="black", linewidths=1.2,  # black outline
            alpha=0.95,
            zorder=10,
            label = "Analysis vmax cyclobs"
        )

        ax.scatter(
            t[mC], cyclobs_arr[mC],
            marker="s", s=55,
            facecolor="#FF0000",                # deep green
            edgecolor="black", linewidths=1.0,
            alpha=0.95,
            zorder=9,
            label ="Vmax cyclobs"
        )

        # --- Optional: custom legend handles (so legend looks perfect)
        handles, labels = ax.get_legend_handles_labels()

        analysis_handle = Line2D(
            [0], [0], marker="*", linestyle="None",
            markerfacecolor="#FFD54F", markeredgecolor="black",
            markeredgewidth=1.2, markersize=12,
            label="Analysis vmax cyclobs"
        )

        cyclobs_handle = Line2D(
            [0], [0], marker="s", linestyle="None",
            markerfacecolor="#2E7D32", markeredgecolor="black",
            markeredgewidth=1.0, markersize=8,
            label="Vmax cyclobs"
        )

        # Put them at the end (or insert wherever you want)
        handles += [analysis_handle, cyclobs_handle]
        ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9)
        ax.set_title(f"Lifecycle Vmax Comparison - 2024013S10093 — RMSE UNET = {rmse:.2f} m/s")
        ax.set_xlabel("Time")
        ax.set_ylabel("Vmax (m/s)")
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





        