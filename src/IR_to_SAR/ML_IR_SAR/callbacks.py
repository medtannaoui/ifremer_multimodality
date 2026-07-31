import os
from pathlib import Path
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

import src.IR_to_SAR.data_preparation.distribution_data_visualisation as distdata
import src.IR_to_SAR.ML_IR_SAR.flow_matching_inference as fm_inf
import src.IR_to_SAR.ML_IR_SAR.callbacks_functions as clbk_func
import importlib
importlib.reload(distdata)
importlib.reload(fm_inf)
importlib.reload(clbk_func)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Arrête l'entraînement quand la métrique surveillée ne s'améliore plus."""

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
    def __init__(self, output_dir, filename="best_regression_model.pt", target_dir=None):
        self.output_path = Path(output_dir) / target_dir / filename
        self.best_val_loss = float("inf")
    def on_validation_epoch_end(self, val_loss, model, fabric, **kwargs):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            fabric.save(self.output_path, {"model": model.state_dict()})
            fabric.print(f"Validation loss improved. Saved model to {self.output_path}")

def _euler_ode(model, z, ir_input, num_steps, device):
    """Intègre l'ODE de Flow Matching par la méthode d'Euler explicite."""
    x_t = z
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((x_t.shape[0],), i / num_steps, device=device)
        model_input = torch.cat([x_t, ir_input], dim=1)
        pred_velocity = model(model_input, t).sample
        x_t = x_t + pred_velocity * dt
    return x_t

class LogValidationSamples:
    def __init__(
        self,
        base_dir,
        mean_X,
        std_X,
        mean_sar,
        std_sar,
        cmap_ir="gray",
        cmap_sar="viridis",
        num_epochs=0,
        infos_train=None,
        infos_val=None,
        infos_test=None,
        mask_train=None,
        mask_val=None,
        mask_test=None,
        target_dir=None,
        cfg=None,
    ):   
        self.cfg = cfg
        self.base_dir = Path(base_dir)
        self.output_dir = target_dir
        self.num_samples = self.cfg.num_val_exemples
        self.start_epoch = self.cfg.start_epoch
        self.every_n_epochs = self.cfg.plot_interval
        self.cmap_ir = cmap_ir
        self.cmap_sar = cmap_sar
        self.mean_sar = mean_sar
        self.mean_X = mean_X
        self.std_sar = std_sar
        self.std_X = std_X
        self.norm = self.cfg.norm
        self.mask_train = mask_train
        self.mask_val = mask_val
        self.mask_test = mask_test
        self.num_epochs = num_epochs
        self.infos_train = infos_train
        self.infos_val = infos_val
        self.infos_test = infos_test
        self.vmax_bins_knots = None
        self.rmax_bins_km = None
        self.output_data = self.cfg.output_data
        self.conditional_model = self.cfg.conditional_model
        self.irwin_channels = self.cfg.irwin_channels
        self.add_era5 = self.cfg.add_era5


    def log_batch(
        self,
        model,
        batch,
        epoch,
        device,
        set="validation",
        ode_pred=None,
        reg_model=None,
        resid_stats=None,
    ):
        model.eval()
        if ode_pred is None:
            x, sar, mask, infos = batch
        else:
            x, sar, mask, infos, ode_pred = batch

        cyclone_id = [d["cyclone_id"] for d in infos]
        sar_time = [d["sar_time"] for d in infos]
        #vmax = [d["vmax"] for d in infos]
        analysis_vmax = [d["analysis_vmax"] for d in infos]
        analysis_rmax = [d["analysis_rmax"] for d in infos]
        x = x.to(device)
        sar = sar.to(device)
        mask = mask.to(device)

        if ode_pred is None:
            with torch.no_grad():
                if not self.conditional_model:
                    if not self.cfg.use_flow_matching:
                        pred = model(x, timestep=0).sample
                    elif not self.cfg.use_residu:
                        B, _, H, W = x.shape
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
                            num_steps=self.cfg.fm_num_inference_steps,
                        )
                        pred = fm_inf.reconstruct_from_residual(
                            resid_norm,
                            mean_pred,
                            resid_stats["mean"],
                            resid_stats["std"],
                        )
                else:
                    shear = torch.stack(
                        [torch.as_tensor(d["shear"], dtype=torch.float32) for d in infos]
                    ).to(device)
                    pred = model(x, timestep=0, cond=shear).sample
        else:
            pred = ode_pred.to(device)
            print("using precomputed ode pred", pred.shape)

        
        ch = x.shape[1] // 4 if x.shape[1] > 4 else 0
        ir = x[:, ch, :, :].squeeze().cpu().numpy()
        sar = sar.squeeze().cpu().numpy() if sar.ndim == 4 else sar.cpu().numpy()
        pred = pred.squeeze().cpu().numpy() if pred.ndim == 4 else pred.cpu().numpy()

        ir_denorm = clbk_func.denorm(ir, self.mean_X[ch], self.std_X[ch])
        if self.norm == "z_score":
            sar_denorm = clbk_func.denorm(sar, self.mean_sar, self.std_sar)
            pred_denorm = clbk_func.denorm(pred, self.mean_sar, self.std_sar)
        elif self.norm == "annular":
            sar_denorm = clbk_func.annular_denormalization(sar, stats={"mean": self.mean_sar, "std": self.std_sar})
            pred_denorm = clbk_func.annular_denormalization(pred, stats={"mean": self.mean_sar, "std": self.std_sar})

        if self.output_data == "aam":
            sar_denorm = clbk_func.moment_to_sar(sar_denorm)
            pred_denorm = clbk_func.moment_to_sar(pred_denorm)

        B, H, W = sar_denorm.shape
        ir_np = ir_denorm
        sar_np = sar_denorm
        pred_np = pred_denorm
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        infos_np = infos.cpu().numpy() if isinstance(infos, torch.Tensor) else infos


        B, H, W = pred_np.shape
        pred_vmax = np.full(B, np.nan, dtype=np.float32)
        pred_rmax_km = np.full(B, np.nan, dtype=np.float32)
        for i in range(B):
            pred_vmax[i], pred_rmax_km[i] = clbk_func.compute_vmax1d_rmax1d(pred_np[i])
        analysis_vmax = np.array(analysis_vmax, dtype=np.float32)
        analysis_rmax = np.array(analysis_rmax, dtype=np.float32)
        analysis_rmax_km = analysis_rmax / 1000.0
        ok_vmax = np.isfinite(analysis_vmax) & np.isfinite(pred_vmax)
        ok_rmax = np.isfinite(analysis_rmax_km) & np.isfinite(pred_rmax_km)
        err_vmax = (pred_vmax - analysis_vmax)[ok_vmax]
        cat_vmax = analysis_vmax[ok_vmax]
        err_rmax = (pred_rmax_km - analysis_rmax_km)[ok_rmax]
        cat_rmax = analysis_rmax_km[ok_rmax]
        if set == "train":
            self.vmax_bins_knots = clbk_func._split_bins_from_train(cat_vmax, n_intervals=3)
            self.rmax_bins_km = clbk_func._split_bins_from_train(cat_rmax, n_intervals=3)
        if self.vmax_bins_knots is None:
            self.vmax_bins_knots = clbk_func._split_bins_from_train(cat_vmax, n_intervals=3)
        if self.rmax_bins_km is None:
            self.rmax_bins_km = clbk_func._split_bins_from_train(cat_rmax, n_intervals=3)
        if self.vmax_bins_knots is not None and err_vmax.size > 0:
            clbk_func._plot_4panel_error_hist(
                errors=err_vmax,
                cat_values=cat_vmax,
                bins=[0, 32, 49, 80],
                title_prefix="Vmax error",
                unit="m/s",
                save_path=os.path.join(self.output_dir, "errors_hist", set, f"vmax_error_epoch_{epoch + 1:04d}.png"),
            )
        if self.rmax_bins_km is not None and err_rmax.size > 0:
            clbk_func._plot_4panel_error_hist(
                errors=err_rmax,
                cat_values=cat_rmax,
                bins=self.rmax_bins_km,
                title_prefix="Rmax error",
                unit="km",
                save_path=os.path.join(self.output_dir, "errors_hist", set, f"rmax_error_epoch_{epoch + 1:04d}.png"),
            )

        os.makedirs(os.path.join(self.output_dir, "samples", set), exist_ok=True)
        sar_all = sar_denorm.flatten()
        pred_all = pred_denorm.flatten()
        mask_all = mask_np.flatten()
        valid = mask_all == 1
        distdata.compare_sar_distribution(
            sar_all[valid], pred_all[valid], self.output_dir, set=set, epoch=epoch
        )
        sar_2d = sar_denorm
        pred_2d = pred_denorm
        mask_2d = mask_np
        os.makedirs(os.path.join(self.output_dir, "predictions_denormalisees", set), exist_ok=True)
        with open(
            os.path.join(self.output_dir, "predictions_denormalisees", set, "predictions_denormalisées.pkl"), "wb"
        ) as f:
            pkl.dump({f"{set}": [ir_np, sar_2d, pred_2d, mask_2d, infos_np], "model": model}, f)
        
        distdata.vmax_compare(analysis_vmax, pred_2d * mask_2d, self.output_dir, set=set, epoch=epoch)


        distdata.rmax_compare(analysis_rmax, pred_2d * mask_2d, self.output_dir, set=set, epoch=epoch)

        distdata.compare_radial_vmax(sar_2d, pred_2d * mask_2d, output_dir=self.output_dir,
                                     set=set, epoch=epoch, plot=False)
        distdata.compute_mae_metric(sar_2d, pred_2d * mask_2d, output_dir=self.output_dir,
                                    set=set, epoch=epoch, plot=False)

        batch_size = ir_np.shape[0]
        num = min(batch_size, self.num_samples)
        np.random.seed(0)
        sample_ids = np.random.choice(batch_size, size=num, replace=False)
        for i in sample_ids:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            H, W = sar_np[i].shape
            rmax = analysis_rmax[i] / 1000
            vmax_i = analysis_vmax[i]
            if vmax_i is None or np.isnan(vmax_i):
                vmax_i = 99999
            if rmax is None or np.isnan(rmax):
                rmax = 99999
            vmax1d_pred, rmax1d_pred = clbk_func.compute_vmax1d_rmax1d(pred_np[i])
            vmax_sar, rmax_sar = clbk_func.compute_vmax1d_rmax1d(sar_np[i])
            distdata.plot_ir(ir_np[i], cmap=self.cmap_ir, ax=axes[0], fig=fig, x_lim=H)
            axes[0].set_title("Infrared Data (t=0) (°C)")
            axes[0].axis("off")
            sar_vis = np.where(mask_np[i] == 1, sar_np[i], np.nan)
            distdata.plot_sar(sar_vis, cmap=self.cmap_sar, ax=axes[1], fig=fig, x_lim=H)
            axes[1].set_title("SAR Observation (m/s)")
            axes[1].axhline(y=0, color="black", linewidth=1)
            axes[1].axvline(x=0, color="black", linewidth=1)
            axes[1].add_patch(Circle((0, 0), radius=rmax, color="black", fill=False, linestyle="--"))
            axes[1].axis("off")
            distdata.plot_sar(pred_np[i], cmap=self.cmap_sar, ax=axes[2], fig=fig, x_lim=H)
            axes[2].set_title("Reconstruction (m/s)")
            axes[2].axhline(y=0, color="black", linewidth=1)
            axes[2].axvline(x=0, color="black", linewidth=1)
            axes[2].add_patch(Circle((0, 0), radius=rmax, color="black", fill=False, linestyle="--"))
            axes[2].axis("off")
            fig.suptitle(
                f"Cyclone: {cyclone_id[i]} — SAR Time: {sar_time[i]} — Epoch {epoch + 1}\n"
                f"Analysis Rmax = {rmax:.1f} Km — Rmax SAR = {rmax_sar:.1f} km — Predicted Rmax1D = {rmax1d_pred:.1f} Km\n"
                f"Analysis Vmax = {vmax_i:.1f} m/s — Vmax SAR = {vmax_sar:.1f} m/s — Predicted Vmax1D = {vmax1d_pred:.1f} m/s",
                fontsize=10,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "samples", set, f"sample_{i}_epoch_{epoch + 1}.png"), dpi=150)
            plt.close(fig)

        print(f"💾 Saved {num} sample images.")

        if self.cfg.use_flow_matching:
            os.makedirs(os.path.join(self.output_dir, f"fm_diagnostics_{set}", "rank_hist_and_samples"), exist_ok=True)
            rank_smple_path = os.path.join(self.output_dir, f"fm_diagnostics_{set}", "rank_hist_and_samples")

            print(f"starting fm diagnostics for {set} set")
            mean_ph, std_ph = [], []
            check_indices = [len(x) // k for k in [1,2,3,4]]  # indices pour lesquels on sauvegarde les samples et rank histogram
            x_fm, sar_fm, mask_fm, infos_fm = x, sar, mask, infos
            for i, (x_ir, sar_target, mask_i, infos_i) in enumerate(zip(x_fm, sar_fm, mask_fm, infos_fm)):
                x_ir = x_ir.unsqueeze(0).to(device)
                sar_target = torch.from_numpy(np.expand_dims(sar_target, axis=0)).to(device)
                mask_i = mask_i.unsqueeze(0).to(device)
                stats = {"mean": self.mean_sar, "std": self.std_sar}
                

                mean_pred = reg_model(x_ir, timestep=0).sample
                ensemble = fm_inf.generate_residual_ensemble(
                    model, mean_pred,
                    residual_mean=resid_stats["mean"],
                    residual_std=resid_stats["std"],
                )
                if i in check_indices:
                    save_dir_sample = os.path.join(rank_smple_path, f"{cyclone_id[i]}_{sar_time[i]}")
                    os.makedirs(save_dir_sample, exist_ok=True)
                    ens_mean_phys, ens_std_phys = fm_inf.plot_ensemble_results(clbk_func,self.cfg.norm,
                        x_ir, sar_target=sar_target, ensemble=ensemble, stats=stats, mask=mask_i,
                        save_path=os.path.join(save_dir_sample, "samples.png"),
                        cmap_sar=self.cmap_sar, cmap_ir=self.cmap_ir, save_pic=True,
                    )
                    ranks, n_members = fm_inf.rank_histogram(ensemble, sar_target, mask_i)
                    fm_inf.plot_rank_histogram(ranks, n_members=n_members,
                                              save_pth=os.path.join(save_dir_sample, "rank_histogram.png"))
                else:
                    ens_mean_phys, ens_std_phys = fm_inf.plot_ensemble_results(clbk_func,self.cfg.norm,
                        x_ir, sar_target=sar_target, ensemble=ensemble, stats=stats, mask=mask_i,
                        save_path=None, cmap_sar=self.cmap_sar, cmap_ir=self.cmap_ir, save_pic=False,
                    )
                mean_ph.append(ens_mean_phys)
                std_ph.append(ens_std_phys)
            mean_ph = np.array(mean_ph)
            std_ph = np.array(std_ph)
            mean_std_dir = os.path.join(self.output_dir, f"fm_diagnostics_{set}", "mean_std_plots")
            os.makedirs(mean_std_dir, exist_ok=True)
            distdata.vmax_compare(analysis_vmax, mean_ph * mask_2d, self.output_dir, set=set, epoch=epoch,
                                  y_label="Vmax Mean ensemble fm (m/s)",
                                  output=os.path.join(mean_std_dir, "mean_vmax.png"))
            distdata.rmax_compare(analysis_rmax, mean_ph * mask_2d, self.output_dir, set=set, epoch=epoch,
                                  y_label="Rmax Mean ensemble fm (km)",
                                  output=os.path.join(mean_std_dir, "mean_rmax_epoch.png"))

            for label, field in [("Mean", mean_ph)]:
                distdata.compare_radial_vmax(sar_2d, field * mask_2d, output_dir=self.output_dir, set=set,
                                             epoch=epoch, plot=False,
                                             output=os.path.join(mean_std_dir, f"{label}_radial_vmax.png"))
                distdata.compute_mae_metric(sar_2d, field * mask_2d, output_dir=self.output_dir, set=set,
                                            epoch=epoch, plot=False,
                                            output=os.path.join(mean_std_dir, f"{label}_mae.png"))
            sar_flat = sar_2d.flatten()
            mask_flat = mask_2d.flatten()
            valid = mask_flat == 1
            distdata.compare_sar_distribution(sar_flat[valid], mean_ph.flatten()[valid],
                                              self.output_dir, set, epoch,
                                              output=os.path.join(mean_std_dir, "mean_distribution.png"))

    def plot_fm_diagnostics(
        self,
        model,
        x_ir,
        sar_target,
        mask,
        stats,
        device="cpu",
        num_steps=20,
        n_rows=5,
        set="validation",
        epoch=0,
        cmap_sar="viridis",
        cmap_ir="gray",
        reg_model=None,
        resid_stats=None,
    ):
        model.eval()
        B = min(x_ir.shape[0], n_rows)
        x_ir = x_ir[:B].to(device)
        sar_tgt = sar_target[:B].to(device)
        mask_v = mask[:B].to(device)

        if sar_tgt.ndim == 3:
            sar_tgt = sar_tgt.unsqueeze(1)
            mask_v = mask_v.unsqueeze(1)

        with torch.no_grad():
            if not self.cfg.use_residu:
                z = torch.randn(B, 1, x_ir.shape[2], x_ir.shape[3], device=device)
                fm_out = _euler_ode(model, z, x_ir, num_steps, device)
                t_mid = torch.full((B,), 0.5, device=device)
                vel05 = model(torch.cat([z, x_ir], dim=1), t_mid).sample
            else:
                z = torch.randn(B, 1, x_ir.shape[2], x_ir.shape[3], device=device)
                mean_pred = reg_model(x_ir, timestep=0).sample
                resid_norm = fm_inf.ode_solver_residual(
                    fm_model=model, z=z, mean_pred=mean_pred,
                    num_steps=self.cfg.fm_num_inference_steps,
                )
                fm_out = fm_inf.reconstruct_from_residual(
                    resid_norm, mean_pred, resid_stats["mean"], resid_stats["std"]
                )
        sar_mean = stats["mean"]
        sar_std = stats["std"]
        fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
        if B == 1:
            axes = axes[None]
        for i in range(B):
            im_ir = x_ir[i, 0].cpu().numpy() * self.std_X[0] + self.mean_X[0]
            if self.cfg.norm == "z_score" : 
                im_tgt = sar_tgt[i, 0].cpu().numpy() * sar_std + sar_mean
                im_fm = fm_out[i, 0].cpu().numpy() * sar_std + sar_mean
                im_vel = (
                        vel05[i, 0].cpu().numpy()
                        if not self.cfg.use_residu
                        else (resid_norm[i, 0] * sar_std + sar_mean).cpu().numpy()
                            )
            elif self.cfg.norm == "annular" : 
                im_tgt = clbk_func.annular_denormalization(sar_tgt[i, 0].cpu().numpy(), stats={"mean":sar_mean,"std":sar_std})
                im_fm = clbk_func.annular_denormalization(fm_out[i, 0].cpu().numpy(), stats={"mean":sar_mean,"std":sar_std})
                im_vel = clbk_func.annular_denormalization(resid_norm[i, 0].cpu().numpy(), stats={"mean":sar_mean,"std":sar_std})
            distdata.plot_ir(im_ir, fig=fig, ax=axes[i, 0], cmap=cmap_ir)
            axes[i, 0].set_title("IR (ch 0)")
            distdata.plot_sar(im_tgt, fig=fig, ax=axes[i, 1], cmap=cmap_sar or "RdBu_r")
            axes[i, 1].set_title("SAR target")
            distdata.plot_sar(im_fm, fig=fig, ax=axes[i, 2], cmap=cmap_sar or "RdBu_r")
            axes[i, 2].set_title("FM sample")
            distdata.plot_sar(im_vel, fig=fig, ax=axes[i, 3], cmap="seismic")
            axes[i, 3].set_title("Velocity @ t=0.5" if not self.cfg.use_residu else "Residu")
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        fm_diag_dir = os.path.join(self.output_dir, f"fm_diagnostics_{set}")
        os.makedirs(fm_diag_dir, exist_ok=True)
        plt.savefig(os.path.join(fm_diag_dir, "fm_diagnostics.png"), dpi=150)
        if self.cfg.use_residu:
            for i in range(B):
                im_ir = x_ir[i, 0].cpu().numpy() * self.std_X[0] + self.mean_X[0]
                if self.cfg.norm == "z_score" : 
                    im_tgt = sar_tgt[i, 0].cpu().numpy() * sar_std + sar_mean
                    im_fm = fm_out[i, 0].cpu().numpy() * sar_std + sar_mean
                elif self.cfg.norm == "annular":
                    im_tgt = clbk_func.annular_denormalization(sar_tgt[i, 0].cpu().numpy(), stats={"mean": sar_mean, "std": sar_std}).squeeze()
                    im_fm = clbk_func.annular_denormalization(fm_out[i, 0].cpu().numpy(), stats={"mean": sar_mean, "std": sar_std}).squeeze()
                reg_pred = reg_model(x_ir[i].unsqueeze(0), timestep=0).sample
                residual_ensemble = fm_inf.generate_residual_ensemble(
                    model, reg_pred, 20,
                    self.cfg.fm_num_inference_steps,
                    resid_stats["mean"],
                    resid_stats["std"],
                ).cpu().numpy()
                fig2 = distdata.plot_comparison(clbk_func, self.cfg.norm, im_ir, im_tgt, mask_v[i, 0].cpu().numpy(),
                                                stats, reg_pred.cpu().numpy(), residual_ensemble)
                plt.savefig(os.path.join(fm_diag_dir, f"plot_comparison_residual_sample_{i}.png"))
                plt.close(fig2)

    def on_validation_plots(self, model, epoch, dataloader, device, reg_model=None, resid_stats=None):
        print(f"📸 Logging samples at epoch {epoch + 1}")

        all_ir, all_sar, all_mask, all_infos = [], [], [], []
        active_set = "test"
        for ir, sar, mask, inf in dataloader[0]:
            all_ir.append(ir)
            all_sar.append(sar)
            all_mask.append(mask)
            all_infos.append(inf)
        ir_full = torch.cat(all_ir, dim=0)
        sar_full = torch.cat(all_sar, dim=0)
        mask_full = torch.cat(all_mask, dim=0)
        infos_full = [d for batch in all_infos for d in batch]
        batch_active = (ir_full, sar_full, mask_full, infos_full)
        self.log_batch(model, batch_active, epoch, device,
                       set=active_set, reg_model=reg_model, resid_stats=resid_stats)

        if  (not self.conditional_model or self.cfg.anggrek_test) :  # 
            ir_anggrek, sar_anggrek, mask_anggrek, infos_anggrek = [], [], [], []
            for ir, sar, mask, inf in dataloader[-1]:
                ir_anggrek.append(ir)
                sar_anggrek.append(sar)
                mask_anggrek.append(mask)
                infos_anggrek.append(inf)
            ir_full_anggrek = torch.cat(ir_anggrek, dim=0)
            sar_full_anggrek = torch.cat(sar_anggrek, dim=0)
            mask_anggrek_full = torch.cat(mask_anggrek, dim=0)
            infos_anggrek_full = [d for batch in infos_anggrek for d in batch]
            batch_anggrek = (ir_full_anggrek, sar_full_anggrek, mask_anggrek_full, infos_anggrek_full)
            self.anggrek_plots(model, batch_anggrek, epoch, device,
                               reg_model=reg_model, resid_stats=resid_stats)
            print("finished anggrek plots")

        if self.cfg.use_flow_matching:
            print(f"starting fm diagnostics for {active_set} set")
            self.plot_fm_diagnostics(
                model, ir_full, sar_full, mask_full,
                stats={"mean": self.mean_sar, "std": self.std_sar},
                device=device,
                num_steps=self.cfg.fm_num_inference_steps,
                n_rows=5 if self.cfg.code_test else 8,
                set=active_set,
                epoch=epoch,
                cmap_sar=self.cmap_sar,
                cmap_ir=self.cmap_ir,
                reg_model=reg_model,
                resid_stats=resid_stats,
            )
            if  epoch > 0 and (not self.conditional_model or self.cfg.anggrek_test):
                self.anggrek_plots(model, batch_anggrek, epoch, device,
                               add_mean_std_fm=True, reg_model=reg_model, resid_stats=resid_stats)

            if not self.cfg.use_residu:
                ode_guidances, ode_reprojections = [], []
                loader_idx = 0 
                for ir, sar, mask, inf in dataloader[loader_idx]:
                    ir = ir.to(device)
                    sar = sar.to(device)
                    mask = mask.to(device)
                    z = torch.randn_like(sar)
                    ode_guidances.append(
                        fm_inf.ode_solver_with_guidance(
                            model=model, z=z, ir_input=ir, obs_sar=sar, obs_mask=mask,
                            num_steps=self.cfg.fm_num_inference_steps,
                            sar_mean_stat=self.mean_sar, sar_std_stat=self.std_sar,
                        )
                    )
                    ode_reprojections.append(
                        fm_inf.ode_solver_with_reprojection(
                            model=model, z=z, ir_input=ir, obs_sar=sar,
                            obs_mask=mask, num_steps=self.cfg.fm_num_inference_steps,
                        )
                    )
                ode_guidances = torch.cat(ode_guidances, dim=0)
                ode_reprojections = torch.cat(ode_reprojections, dim=0)
                self.log_batch(model,
                               (*batch_active, ode_guidances),
                               epoch, device,
                               set=f"{active_set}_guidance",
                               ode_pred=True,
                               reg_model=reg_model, resid_stats=resid_stats)
                self.log_batch(model,
                               (*batch_active, ode_reprojections),
                               epoch, device,
                               set=f"{active_set}_reprojection",
                               ode_pred=True,
                               reg_model=reg_model, resid_stats=resid_stats)


    def anggrek_plots(
        self, model, batch, epoch, device,
        add_mean_std_fm=False, reg_model=None, resid_stats=None
    ):
        model.eval()
        x, _, _, infos = batch
        x = x.to(device)

        def moment_to_sar(moment):
            assert moment.ndim == 3
            _, H, W = moment.shape
            y, x_idx = np.indices((H, W))
            cy, cx = H // 2, W // 2
            r = np.sqrt((x_idx - cx) ** 2 + (y - cy) ** 2)
            return moment / np.maximum(r, 1.0)[None, :, :]

        with torch.no_grad():
            if not add_mean_std_fm:
                if not self.conditional_model:
                    if not self.cfg.use_flow_matching:
                        pred = model(x, timestep=0).sample
                    elif not self.cfg.use_residu:
                        B, _, H, W = x.shape
                        z = torch.randn(B, 1, H, W, device=device)
                        pred = _euler_ode(model, z, x, self.cfg.fm_num_inference_steps, device)
                    else:
                        B, _, H, W = x.shape
                        z = torch.randn(B, 1, H, W, device=device)
                        mean_pred = reg_model(x, timestep=0).sample
                        resid_norm = fm_inf.ode_solver_residual(
                            fm_model=model, z=z, mean_pred=mean_pred,
                            num_steps=self.cfg.fm_num_inference_steps,
                        )
                        pred = fm_inf.reconstruct_from_residual(
                            resid_norm, mean_pred, resid_stats["mean"], resid_stats["std"]
                        )
                else:
                    shear = torch.stack(
                        [torch.as_tensor(d["shear"], dtype=torch.float32) for d in infos]
                    ).to(device)
                    pred = model(x, timestep=0, cond=shear).sample
        
        sar_time = [d.get("date") for d in infos]
        time_parsed = pd.to_datetime(sar_time, errors="coerce")
        order = np.argsort(time_parsed.values.astype("datetime64[ns]"))
        time_parsed = time_parsed[order]
        infos_ord = [infos[i] for i in order]

        if not add_mean_std_fm:
            print("fm diagnostics for anggrek ......")
            x_np = x.detach().cpu().numpy()
            pred_np = pred.detach().squeeze().cpu().numpy() if pred.ndim == 4 else pred.detach().cpu().numpy()
            if pred_np.ndim == 2:
                pred_np = pred_np[None, ...]
            B = x_np.shape[0]
            ch = 4 if x_np.shape[1] > 4 else 0
            ir = x_np[:, ch, :, :]
            if self.add_era5:
                era5 = x_np[:, -1, :, :]
                era5 = clbk_func.denorm(era5, self.mean_X[-1], self.std_X[-1])
            ir_den = clbk_func.denorm(ir, self.mean_X[ch], self.std_X[ch])
            if self.norm == "z_score":
                pred_den = clbk_func.denorm(pred_np, self.mean_sar, self.std_sar)
            elif self.norm == "annular":
                pred_den = clbk_func.annular_denormalization(pred_np, stats={"mean": self.mean_sar, "std": self.std_sar})
            else:
                pred_den = pred_np

            if self.output_data == "aam":
                pred_den = moment_to_sar(pred_den)

            ir_den = ir_den[order]
            pred_den = pred_den[order]

            os.makedirs(os.path.join(self.output_dir, "model"), exist_ok=True)
            with open(os.path.join(self.output_dir, "model",
                                   "weights.pkl"), "wb") as f:
                pkl.dump({"model": model}, f)

            out_root = Path(self.output_dir) / "anggrek_monitoring"
            field_dir = out_root / "field_plots"
            field_dir.mkdir(parents=True, exist_ok=True)

        else:
            os.makedirs(os.path.join(self.output_dir, "fm_diagnostics_anggrek", "anggrek_monitoring"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "fm_diagnostics_anggrek", "prediction_denormalisées"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "fm_diagnostics_anggrek", "anggrek_monitoring", "fields_plots"), exist_ok=True)
            out_root = Path(self.output_dir) / "fm_diagnostics_anggrek" / "anggrek_monitoring"

            mean_ph, std_ph = [], []
            for i, x_fm in enumerate(x):
                if x_fm.ndim == 3:
                    x_fm = x_fm.unsqueeze(0)
                if not self.cfg.use_residu:
                    ensemble = fm_inf.generate_ensemble(model=model, ir_input=x_fm, device=device)
                else:
                    mean_pred = reg_model(x_fm, timestep=0).sample
                    ensemble = fm_inf.generate_residual_ensemble(
                        model, mean_pred,
                        num_steps=self.cfg.fm_num_inference_steps,
                        residual_mean=resid_stats["mean"],
                        residual_std=resid_stats["std"],
                    )
                mean_fm, std_fm = fm_inf.plot_ensemble_results(clbk_func,self.cfg.norm,
                    x_fm, sar_target=None, ensemble=ensemble,
                    stats={"mean": self.mean_sar, "std": self.std_sar},
                    mask=None,
                    save_path=os.path.join(
                        self.output_dir, "fm_diagnostics_anggrek", "anggrek_monitoring",
                        "fields_plots", f"{time_parsed[i].strftime('%Y%m%d%H%M%S')}_ensemble.png"
                    ),
                    cmap_sar=self.cmap_sar, cmap_ir=self.cmap_ir, save_pic=True,
                )
                mean_ph.append(mean_fm)
                std_ph.append(std_fm)

            mean_ph = np.array(mean_ph)
            std_ph = np.array(std_ph)
            x_np = x.detach().cpu().numpy()
            with open(os.path.join(self.output_dir, "fm_diagnostics_anggrek",
                                   "prediction_denormalisées", "mean_std_fm.pkl"), "wb") as f:
                pkl.dump({"mean_fm": mean_ph, "std_fm": std_ph, "time": time_parsed,
                          "ir_norm": x_np, "model": model}, f)

        @torch.no_grad()
        def resample_2km_to_3km_torch(xs):
            resamples = []
            for x_i in xs:
                x_t = torch.as_tensor(x_i, dtype=torch.float32, device="cpu")
                if x_t.dim() == 2:
                    x_t = x_t.unsqueeze(0).unsqueeze(0)
                H, W = x_t.shape[-2:]
                in_res = 2
                x3 = F.interpolate(x_t, size=(int(H * in_res / 3), int(W * in_res / 3)),
                                   mode="bilinear", align_corners=False)
                resamples.append(x3.squeeze(0).squeeze(0).cpu().numpy())
            return np.stack(resamples, axis=0)

        if not add_mean_std_fm:
            pred_denorm_3m = resample_2km_to_3km_torch(pred_den)
            pred_vmax = np.nanmax(pred_denorm_3m.reshape(len(pred_den), -1), axis=1)
            pred_vmax_2km = np.nanmax(pred_den.reshape(len(pred_den), -1), axis=1)
        else:
            mean_fm_3m = resample_2km_to_3km_torch(mean_ph)
            std_fm_3m = resample_2km_to_3km_torch(std_ph)
            pred_vmax_mean_fm_3m = np.nanmax(mean_fm_3m.reshape(len(mean_ph), -1), axis=1)
            pred_vmax_std_fm_3m = np.nanmax(std_fm_3m.reshape(len(std_ph), -1), axis=1)

        vmax = np.array([d.get("vmax", np.nan) for d in infos_ord], dtype=float)
        vmax_cyclobs = np.array([d.get("vmax_cyclobs", np.nan) for d in infos_ord], dtype=float)
        analysis_vmax_cyclobs = np.array([d.get("analysis_vmax_cyclobs", np.nan) for d in infos_ord], dtype=float)
        ibtracs_vmax = np.array([d.get("ibtracs_vmax", np.nan) for d in infos_ord], dtype=float)
        satcon_vmax = np.array([d.get("satcon_vmax", np.nan) for d in infos_ord], dtype=float)
        era5_vmaxs = np.array([d.get("era5_vmax", np.nan) for d in infos_ord], dtype=float)

        if not add_mean_std_fm:
            for i in range(B):
                t = time_parsed[i]
                vmax_pred, rmax_pred = clbk_func.compute_vmax1d_rmax1d(pred_den[i])
                date_key = t.strftime("%Y%m%d%H%M%S") if not pd.isna(t) else f"unknown_{i:03d}"
                fname = f"{date_key}_fields.png"
                supt = (
                    f"{t.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Best Track Vmax: {vmax[i]:.2f} m/s | "
                    f"Pred Vmax: {vmax_pred:.2f} m/s | "
                    f"Pred RMW: {rmax_pred:.1f} km"
                ) if not pd.isna(t) else f"Unknown time (idx={i})"

                sub = field_dir
                sub.mkdir(parents=True, exist_ok=True)
                n_panels = 3 if self.add_era5 else 2
                fig, axs = plt.subplots(1, n_panels, figsize=(8, 4), constrained_layout=True)

                distdata.plot_ir(ir_den[i], cmap=self.cmap_ir, ax=axs[0], fig=fig, x_lim=pred_den[i].shape[0])
                axs[0].set_title("IRWIN")
                axs[0].axis("off")

                pred_ax_idx = 2 if self.add_era5 else 1
                distdata.plot_sar(pred_den[i], cmap=self.cmap_sar, ax=axs[pred_ax_idx], fig=fig,
                                  x_lim=pred_den[i].shape[0])
                axs[pred_ax_idx].set_title("Prediction")
                axs[pred_ax_idx].axhline(y=0, color="black", linewidth=1)
                axs[pred_ax_idx].axvline(x=0, color="black", linewidth=1)
                axs[pred_ax_idx].add_patch(
                    Circle((0, 0), radius=rmax_pred, color="black", fill=False, linestyle="--")
                )

                if self.add_era5:
                    era5_vis = era5[i]
                    
                    distdata.plot_sar(era5_vis, cmap=self.cmap_sar, ax=axs[1], fig=fig,
                                      x_lim=pred_den[i].shape[0])
                    axs[1].set_title("ERA5")
                    axs[1].axis("off")

                fig.suptitle(supt)
                fig.savefig(sub / fname, dpi=150)
                plt.close(fig)

        if not add_mean_std_fm:
            ok = np.isfinite(vmax) & np.isfinite(pred_vmax)
            rmse = float(np.sqrt(np.mean((pred_vmax[ok] - vmax[ok]) ** 2))) if np.any(ok) else np.nan
        else:
            ok = np.isfinite(vmax) & np.isfinite(pred_vmax_mean_fm_3m)
            rmse = float(np.sqrt(np.mean((pred_vmax_mean_fm_3m[ok] - vmax[ok]) ** 2))) if np.any(ok) else np.nan

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(time_parsed, ibtracs_vmax, color="black", linewidth=2, label="IBTrACS (Best Track)")
        # ax.plot(time_parsed, vmax, color="red", linewidth=2, label="ATCF")
        # ax.plot(time_parsed, satcon_vmax, color="blue", linewidth=2, label="SATCON")
        ax.plot(time_parsed, era5_vmaxs, color="orange", linewidth=2, label="ERA5")

        if not add_mean_std_fm:
            ax.plot(time_parsed, pred_vmax, color="green", linewidth=2, label="UNET Res 3 km")
            # ax.plot(time_parsed, pred_vmax_2km, color="magenta", linewidth=2, label="UNET Res 2 km")
        else:
            ax.plot(time_parsed, pred_vmax_mean_fm_3m, color="green", linewidth=2,
                    label="flow matching (mean n=20) Res 3 km")
            ax.fill_between(
                time_parsed,
                pred_vmax_mean_fm_3m - pred_vmax_std_fm_3m,
                pred_vmax_mean_fm_3m + pred_vmax_std_fm_3m,
                color="magenta", alpha=0.10, label="Residual FM ± std",
            )
        t_arr = np.array(time_parsed)
        mA = ~np.isnan(analysis_vmax_cyclobs)
        mC = ~np.isnan(vmax_cyclobs)
        ax.scatter(t_arr[mA], analysis_vmax_cyclobs[mA], marker="*", s=110,
                   facecolor="#FFD54F", edgecolor="black", linewidths=1.2, alpha=0.95, zorder=10,
                   label="Analysis vmax cyclobs")
        # ax.scatter(t_arr[mC], vmax_cyclobs[mC], marker="s", s=55,
        #            facecolor="#FF0000", edgecolor="black", linewidths=1.0, alpha=0.95, zorder=9,
        #            label="Vmax cyclobs")
        handles, _ = ax.get_legend_handles_labels()
        handles += [
            Line2D([0], [0], marker="*", linestyle="None", markerfacecolor="#FFD54F",
                   markeredgecolor="black", markeredgewidth=1.2, markersize=12, label="Analysis vmax cyclobs"),
            # Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="#2E7D32",
            #        markeredgecolor="black", markeredgewidth=1.0, markersize=8, label="Vmax cyclobs"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9)
        ax.set_title(f"Lifecycle Vmax Comparison - 2024013S10093 — RMSE UNET = {rmse:.2f} m/s")
        ax.set_xlabel("Time")
        ax.set_ylabel("Vmax (m/s)")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_root / "vmax_comparison.png", dpi=150)
        plt.close(fig)