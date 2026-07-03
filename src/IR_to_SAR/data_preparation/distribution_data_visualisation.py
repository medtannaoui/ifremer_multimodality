import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from src.visualisation.utils_colormap import CMAP
from loguru import logger
import shutil

cmap_ir = CMAP.cira_ir()
cmap_sar = CMAP.cmap_sar()



def plot_sar(tensor, fig=None, ax=None, cmap=cmap_sar, title=None,
             x_lim=300, x=None, y=None):
    
    if len(tensor.shape) == 3:
        tensor = tensor.squeeze()
    if ax is None:
        ax = plt.gca()
    if x is None:
        y_sar = np.linspace(-x_lim, x_lim, tensor.shape[0])
        x_sar = np.linspace(-x_lim, x_lim, tensor.shape[1])
        x_sar, y_sar = np.meshgrid(x_sar, y_sar)
    else:
        x_sar, y_sar = x, y   # déjà 2D, pas de meshgrid
    im = ax.pcolormesh(
        x_sar,
        y_sar,
        tensor,
        cmap=cmap,
        vmin=0,
        vmax=160/1.94384449,
        shading="auto"
    )
    if x is None:
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-x_lim, x_lim)
    if title is not None:
        ax.set_title(title)
    if fig is not None:
        fig.colorbar(im, ax=ax, orientation="horizontal")
    ax.set_aspect("equal")

def plot_aam(tensor, fig=None, ax=None,title=None,x_lim=300):
    if len(tensor.shape) == 3:
        tensor = tensor.squeeze()
    if ax is None:
        ax = plt.gca()
    x_sar,y_sar = np.linspace(-x_lim,x_lim,tensor.shape[0]) , np.linspace(-x_lim,x_lim,tensor.shape[1])
    im = ax.pcolormesh(x_sar, y_sar, tensor,vmin=-0,vmax=4500)
    ax.set_xlim(-x_lim,x_lim)
    ax.set_ylim(-x_lim,x_lim)
    if title is not None:
        ax.set_title(title)
    if fig is not None:
        fig.colorbar(im,ax=ax,orientation="horizontal")
    ax.set_aspect('equal')

def plot_ir(tensor, fig=None, x=None, y=None, ax=None, x_lim=300,
            cmap=cmap_ir, vmin=-100, vmax=50):
    if len(tensor.shape) == 3:
        tensor = tensor.squeeze()
    if ax is None:
        ax = plt.gca()
    tensor = np.squeeze(tensor)
    ny, nx = tensor.shape  # ny = rows, nx = cols
    if x is None:
        x = np.linspace(-x_lim, x_lim, nx)
    if y is None:
        y = np.linspace(-x_lim, x_lim, ny)
    im = ax.pcolormesh(
        x, y, tensor,
        cmap=cmap,
        shading="nearest",
        vmin=vmin,
        vmax=vmax
    )
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-x_lim, x_lim)
    ax.set_aspect("equal")
    if fig is not None:
        fig.colorbar(im, ax=ax, orientation="horizontal")

def plot_mw(tensor, cmap="cividis", x=None, y=None, ax=None, x_lim=300,fig=None):
    if ax is None:
        ax = plt.gca()
    tensor = np.array(tensor)
    if x is None:
        x_mw = np.linspace(-300, 300, tensor.shape[1])
        y_mw = np.linspace(-300, 300, tensor.shape[0])
        im = ax.pcolormesh(x_mw, y_mw, tensor, cmap=cmap, shading="nearest")
    else:
        im = ax.pcolormesh(x, y, tensor, cmap=cmap, shading="nearest")
    if fig is not None:
        fig.colorbar(im,ax=ax,orientation="horizontal")
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-x_lim, x_lim)
    ax.set_aspect('equal')
    
def vmax_compare(analysis_vmax, predict_sars, output_dir, set, epoch, plot=False,min=None, max=None, y_label = None, output = None):
    title_end = "Categorie_1" if min==19 and max==63 else "Categorie_2" if min==63 and max==83 else "Categorie_3" if min==83 and max==96 else "Categorie_4" if min==96 and max==113 else "Categorie_5" if min==113 else ""
    vmax_true = []
    vmax_pred = []
    for ana_vmax, sar2 in zip(analysis_vmax, predict_sars):
        if min is None or max is None:
            if ana_vmax is None or np.isnan(ana_vmax):
                continue
        else :    
            if ana_vmax is None or np.isnan(ana_vmax) or  ana_vmax*1.94384 < min or ana_vmax*1.94384 > max:
                continue
        vmax_true.append(ana_vmax)   #m/s
        vmax_pred.append(np.nanmax(sar2)) #m/s
    vmax_true = np.array(vmax_true)
    vmax_pred = np.array(vmax_pred)
    errors = vmax_pred - vmax_true
    mae = np.nanmean(np.abs(errors))
    rmse = np.sqrt(np.nanmean(errors**2))
    bias = np.nanmean(errors)
    coef = np.polyfit(vmax_true, vmax_pred, 1)
    x_line = np.array([vmax_true.min(), vmax_true.max()])
    y_line = coef[0] * x_line + coef[1]
    xy = np.vstack([vmax_true, vmax_pred])
    z = gaussian_kde(xy)(xy)
    fig, axes = plt.subplots(2, 1, figsize=(6, 9), gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax_hist = axes[1]
    sc = ax.scatter(vmax_true, vmax_pred, c=z, cmap="viridis", s=20)
    fig.colorbar(sc, ax=ax, label="Density")
    ax.plot(x_line, x_line, "r--", lw=2, label="Perfect prediction")
    ax.plot(x_line, y_line, "b-", lw=2,
            label=f"Regression: y={coef[0]:.2f}x+{coef[1]:.2f}")
    ax.set_xlabel(f"Analysis Vmax (m/s)")
    ax.set_ylabel(y_label or "Predicted Vmax (m/s)")
    ax.set_title(f"Vmax Comparison — Epoch {epoch+1} ({set}) {title_end}")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    ax_hist.hist(errors, bins=40, color="gray", alpha=0.8)
    ax_hist.set_title("Prediction Error Distribution (Pred - Analysis_vmax)")
    ax_hist.set_xlabel("Prediction Error (m/s)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, linestyle="--", alpha=0.3)
    textstr = (
        f"MAE  : {mae:.2f} m/s\n"
        f"RMSE : {rmse:.2f} m/s\n"
        f"Bias : {bias:.2f} m/s\n"
        f"Max analysis vmax : {vmax_true.max():.2f} m/s\n"
        f"Max predicted vmax: {vmax_pred.max():.2f} m/s"
    )
    ax.text(0.02, 0.98, textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout()
    filename = f"Vmax_Comparison_{title_end}.png"
    if output is None:
        os.makedirs(os.path.join(output_dir,"vmax_compare",set),exist_ok=True)
    plt.savefig(output if output is not None else os.path.join(output_dir,"vmax_compare",set, filename), dpi=150)
    plt.close(fig)

def compare_sar_distribution(sar_knots, pred_knots, output_dir, set, epoch,output=None):
        if set == "train":
            return None
        plt.figure(figsize=(8, 6))
        plt.hist(sar_knots, bins=60, alpha=0.5, density=False, label="True SAR", color="blue")
        plt.hist(pred_knots, bins=60, alpha=0.5, density=False, label="Predicted SAR", color="orange")
        plt.legend()
        plt.xlabel("Wind Speed (m/s)")
        plt.ylabel("Count")
        plt.title(f"Global Wind Speed Distribution —_{set}")
        if output is None:
            os.makedirs(os.path.join(output_dir, "compare_sar_distribution",set),exist_ok=True)
            out_path = os.path.join(output_dir, "compare_sar_distribution",set,f"wind_distribution.png")
            plt.savefig(out_path, dpi=150)
            plt.close('all')
        else:
            plt.savefig(output, dpi=150)
            plt.close('all')
        print(f"📈 Saved global distribution")

def compare_radial_vmax(
    sars_true,      #m/s
    sars_predict,   #m/s
    output_dir, 
    set, 
    epoch,  
    center=None, 
    dr=1, 
    output=None,
    plot=False
):
    assert sars_true.shape == sars_predict.shape, "sars_true and sars_predict must have the same shape"
    N, H, W = sars_true.shape
    if center is None:
        yc, xc = H // 2, W // 2
    else:
        yc, xc = center
    Y, X = np.indices((H, W))
    R = np.sqrt((X - xc)**2 + (Y - yc)**2)
    Rmax = int(R.max())
    r_bins = np.arange(0, Rmax + dr, dr)
    vmax_r_true = []
    vmax_r_pred = []
    err_of_means = []      # RED
    mean_of_errors = []    # ORANGE
    r_centers = []
    for r0 in r_bins[:-1]:
        mask = (R >= r0) & (R < r0 + dr)
        if np.sum(mask) == 0:
            continue
        vmax_true_per_sample = np.nanmax(sars_true[:, mask], axis=1)
        vmax_pred_per_sample = np.nanmax(sars_predict[:, mask], axis=1)
        vmax_true_shell = np.nanmean(vmax_true_per_sample)
        vmax_pred_shell = np.nanmean(vmax_pred_per_sample)
        vmax_r_true.append(vmax_true_shell)
        vmax_r_pred.append(vmax_pred_shell)
        err_of_means.append(np.abs(vmax_true_shell - vmax_pred_shell))
        per_sample_abs_err = np.abs(vmax_true_per_sample - vmax_pred_per_sample)
        mean_of_errors.append(np.nanmean(per_sample_abs_err))
        r_centers.append(r0 + dr / 2)
    vmax_r_true     = np.array(vmax_r_true)
    vmax_r_pred     = np.array(vmax_r_pred)
    err_of_means    = np.array(err_of_means)
    mean_of_errors  = np.array(mean_of_errors)
    r_centers       = np.array(r_centers)
    vmax1d_true = np.nanmax(vmax_r_true)
    vmax1d_pred = np.nanmax(vmax_r_pred)
    rmax1d_true = r_centers[np.nanargmax(vmax_r_true)]
    rmax1d_pred = r_centers[np.nanargmax(vmax_r_pred)]
    r_norm = r_centers /rmax1d_true
    rmax_1d_with_norme = rmax1d_true
    rmax1d_true /= (rmax_1d_with_norme*2)
    rmax1d_pred /= (rmax_1d_with_norme*2)
    error = np.abs(vmax_r_true - vmax_r_pred)
    plt.figure(figsize=(10, 6))
    plt.plot(r_norm, vmax_r_true*1.94384, color="green", linewidth=2, label="Reel SAR Radial Vmax (knots)")
    plt.plot(r_norm, vmax_r_pred*1.94384, color="blue", linewidth=2, label="Predict SAR Radial Vmax (knots)")
    plt.plot(r_norm, err_of_means*1.94384, linestyle="--", linewidth=2,
         label="|Error of means| = |E(true)-E(pred)| (m/s)")
    plt.plot(r_norm, mean_of_errors*1.94384, linestyle="-.", linewidth=2, color="orange",
         label="Mean absolute error = E(|true-pred|) (m/s)")
    plt.axvline(rmax1d_true*2, color="green", linestyle="--", linewidth=1.5, label=f"Reel Rmax1D = {rmax1d_true*2:.1f}")
    plt.axvline(rmax1d_pred*2, color="blue", linestyle="--", linewidth=1.5, label=f"Pred Rmax1D = {rmax1d_pred*2:.1f}")
    plt.axhline(vmax1d_true*1.94384, color="green", linestyle="--", linewidth=1.5, label=f"Reel Vmax1D = {vmax1d_true*1.94384:.1f} m/s")
    plt.axhline(vmax1d_pred*1.94384, color="blue", linestyle="--", linewidth=1.5, label=f"Pred Vmax1D = {vmax1d_pred*1.94384:.1f} m/s")
    plt.xlabel("Radius R* (R/RMW)", fontsize=14)
    plt.ylabel("Vmax Mean (m/s)", fontsize=14)
    plt.title(f"Radial Vmax Profile — {set} (Epoch {epoch+1})", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if not plot:
        os.makedirs(os.path.join(output_dir,"compare_radial_vmax",set),exist_ok=True)
        if output is None:
            os.makedirs(os.path.join(output_dir,"compare_radial_vmax",set),exist_ok=True)
            out_path = os.path.join(output_dir,"compare_radial_vmax",set, f"radial_vmax_{set}.png")
            plt.savefig(out_path, dpi=150)
        else : 
            plt.savefig(output, dpi=150)
        
        plt.close()
    else:
        plt.show()

    return {
        "vmax1d_true": float(vmax1d_true),
        "vmax1d_pred": float(vmax1d_pred),
        "rmax1d_true": float(rmax1d_true),
        "rmax1d_pred": float(rmax1d_pred),
        "radii": r_centers,
        "vmax_profile_true": vmax_r_true,
        "vmax_profile_pred": vmax_r_pred,
    }



def compute_mae_metric(sar_true, sar_pred, output_dir, set, epoch, plot=False, output=None):
    if set=="train":
        return None
    mask = np.isfinite(sar_true)
    mae_global = np.nanmean(np.abs(sar_pred - sar_true))
    mae_map = np.nanmean(np.abs(sar_pred - sar_true), axis=0)  # (H, W)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(mae_map, cmap="inferno")
    plt.colorbar(im, label="MAE (m/s)")
    plt.title(f"Mean Absolute Error Map — {set} (Epoch {epoch+1})")
    plt.axis("off")
    if not plot : 
        if output is None:
            os.makedirs(os.path.join(output_dir,"compute_mae_metric",set),exist_ok=True)
            save_path = os.path.join(output_dir,"compute_mae_metric",set, f"mae_map_{set}.png")
            plt.savefig(save_path, dpi=150)
        else :
            plt.savefig(output, dpi=150)
        plt.close()
    else : 
        plt.show()


def rmax_compare(analysis_rmax, predict_sars, output_dir, set, epoch, plot=False, min=None, max=None, y_label=None, output=None):
    if set == "train":
        return None
    N, H, W = predict_sars.shape
    cx, cy = W // 2, H // 2
    Y, X = np.indices((H, W))
    dist_map = np.sqrt((X - cx)**2 + (Y - cy)**2)
    rmax_true = []
    rmax_pred = []
    title_end = "Categorie_1" if min==0 and max==30 else "Categorie_2" if min==30 and max==60 else "Categorie_3" if min==60 else ""
    for ana_rmax, sar_pred in zip(analysis_rmax, predict_sars):
        if min is None or max is None:
            if ana_rmax is None or np.isnan(ana_rmax):
                continue
        else :
            if ana_rmax is None or np.isnan(ana_rmax) or ana_rmax < min*1000 or ana_rmax > max*1000:
                continue
        rmax_true.append(ana_rmax / 1000.0)
        vmax_p = np.nanmax(sar_pred)
        mask = sar_pred == vmax_p
        rmax_p = np.nanmean(dist_map[mask]) * 2.0  # pixel → km
        rmax_pred.append(rmax_p)
    rmax_true = np.array(rmax_true)
    rmax_pred = np.array(rmax_pred)
    valid = ~np.isnan(rmax_true) & ~np.isnan(rmax_pred)
    rmax_true = rmax_true[valid]
    rmax_pred = rmax_pred[valid]
    errors = rmax_pred - rmax_true
    MAE = np.nanmean(np.abs(errors))
    RMSE = np.sqrt(np.nanmean(errors**2))
    Bias = np.nanmean(errors)
    max_true = np.nanmax(rmax_true)
    max_pred = np.nanmax(rmax_pred)
    coef = np.polyfit(rmax_true, rmax_pred, 1)
    x_line = np.array([rmax_true.min(), rmax_true.max()])
    y_line = coef[0] * x_line + coef[1]
    xy = np.vstack([rmax_true, rmax_pred])
    z = gaussian_kde(xy)(xy)
    fig = plt.figure(figsize=(7, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    sc = ax1.scatter(rmax_true, rmax_pred, c=z, cmap="viridis", s=25)
    plt.colorbar(sc, ax=ax1, label="Density")
    ax1.plot(x_line, x_line, 'r--', linewidth=2, label="Perfect prediction (y=x)")
    ax1.plot(
        x_line, y_line, 'b-', linewidth=2,
        label=f"Regression: y={coef[0]:.2f}x+{coef[1]:.2f}"
    )
    ax1.legend(loc="lower right")

    ax1.set_title(f"Rmax Comparison — Epoch {epoch+1} ({set}) {title_end}", fontsize=15)
    ax1.set_xlabel("Analysis Rmax (km)")
    ax1.set_ylabel(y_label or "Predicted Rmax (km)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    textstr = (
        f"MAE : {MAE:.2f} km\n"
        f"RMSE : {RMSE:.2f} km\n"
        f"Bias : {Bias:.2f} km\n"
        f"Max Analysis Rmax : {max_true:.1f} km\n"
        f"Max Predicted Rmax : {max_pred:.1f} km"
    )
    ax1.text(
        0.05, 0.95, textstr,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.hist(errors, bins=30, color="gray", alpha=0.85)
    ax2.set_title("Prediction Error Distribution (Pred - Analysis Rmax)")
    ax2.set_xlabel("Error (km)")
    ax2.set_ylabel("Count")
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if plot:
        plt.show()
    else:
        if output is None:
            os.makedirs(os.path.join(output_dir,"rmax_compare",set,),exist_ok=True)
        out_path = output if output is not None else os.path.join(output_dir,"rmax_compare",set, f"Rmax_Comparison_{set}_{title_end}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
    return {
        "rmax_true": rmax_true,
        "rmax_pred": rmax_pred,
        "MAE": MAE,
        "RMSE": RMSE,
        "Bias": Bias,
        "regression_coef": coef, 
    }

    

def plot_comparison(
                    clbk,
                    norm,
                    ir_input, 
                    sar_target, 
                    mask, 
                    stats,
                    reg_pred,
                    residual_ensemble,
                    title=""
                    ):

    sar_mean = stats["mean"]
    sar_std  = stats["std"]
    
    def stdmap(ens):
        return ens.std(0).squeeze() * sar_std  # std scales with std
    # Apply mask
    valid_np = (mask.squeeze() > 0) if mask is not None else None
    def apply_mask(arr):
        if valid_np is not None:
            return np.where(valid_np, arr, np.nan)
        return arr
    tgt_np   = apply_mask(sar_target)
    if norm == "z_score":
        reg_np   = reg_pred*sar_std + sar_mean
        res_np   = residual_ensemble.mean(0)*sar_std + sar_mean
    elif norm == "annular":
        reg_np = clbk.annular_denormalization(reg_pred.squeeze(), stats={"mean":sar_mean, "std": sar_std})
        res_np = clbk.annular_denormalization(residual_ensemble.mean(0), stats={"mean":sar_mean, "std": sar_std})

    # res_std  = stdmap(residual_ensemble)
    ir_np    = ir_input.squeeze()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plot_ir(ir_np, fig=fig,ax=axes[0], cmap=cmap_ir);      axes[0].set_title("IR (ch 0)")
    plot_sar(tgt_np, fig=fig,ax=axes[1], cmap=cmap_sar);      axes[1].set_title("SAR target (m/s)")
    plot_sar(reg_np, fig=fig,ax=axes[2], cmap=cmap_sar);      axes[2].set_title("Deterministic regression")
    plot_sar(res_np, fig=fig,ax=axes[3], cmap=cmap_sar);  axes[3].set_title("Residual FM mean")
    if title: fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig

def training_completed(cfg, train_loss_history, val_loss_history,pix2pix_loss_history, gradient_loss_history,radial_loss_history,  target_dir):
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
    history_df = pd.DataFrame({
    "train_loss": train_loss_history,
    "val_loss": val_loss_history,
    "pix2pix_history": pix2pix_loss_history,
    "gradient_loss_history" : gradient_loss_history,
    "radial_loss_history" : radial_loss_history
    })
    csv_path = target_dir / "training_history.csv" 
    history_df.to_csv(csv_path, index=False)
    pix_train = [x[0] for x in pix2pix_loss_history]
    pix_val   = [x[1] for x in pix2pix_loss_history]
    grad_train = [x[0] for x in gradient_loss_history]
    grad_val   = [x[1] for x in gradient_loss_history]
    rad_train = [x[0] for x in radial_loss_history]
    rad_val   = [x[1] for x in radial_loss_history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    axes[0].plot(pix_train, label="Train")
    axes[0].plot(pix_val, label="Val")
    axes[0].set_title("Pix2Pix Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(grad_train, label="Train")
    axes[1].plot(grad_val, label="Val")
    axes[1].set_title("Gradient Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)
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
    dst_cfg = os.path.join(target_dir, "config.yaml")
    config_path ="/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/config.yaml"
    if hasattr(cfg, "config_path") and config_path and os.path.exists(config_path):
        shutil.copy(config_path, dst_cfg)
        logger.info(f"📄 Config copied to: {dst_cfg}")
    else:
        with open(os.path.join(target_dir, "config_fallback.txt"), "w") as f:
            for k, v in cfg.__dict__.items():
                f.write(f"{k}: {v}\n")
        logger.warning("⚠️ config.yaml path not found, saved config_fallback.txt instead")


def plot_metric_scatter(
    true_values,            # liste ou array des valeurs vraies
    pred_values,            # liste ou array des valeurs prédites
    output_path,            # chemin complet fichier .png
    file_name,
    title="Metric Comparison",
    xlabel="True Values",
    ylabel="Predicted Values",
    stats_title="Statistics"
):
    true_values = np.array(true_values, dtype=float)
    pred_values = np.array(pred_values, dtype=float)
    errors = pred_values - true_values
    mae = np.nanmean(np.abs(errors))
    rmse = np.sqrt(np.nanmean(errors**2))
    bias = np.nanmean(errors)
    textstr = (
        f"MAE  : {mae:.2f} m\s\n"
        f"RMSE : {rmse:.2f} m\s\n"
        f"Bias : {bias:.2f} m\s\n"
    )
    plt.figure(figsize=(7, 7))
    plt.scatter(true_values, pred_values, alpha=0.5, color="#1f77b4", edgecolors="none")
    min_v = min(true_values.min(), pred_values.min())
    max_v = max(true_values.max(), pred_values.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    ax = plt.gca()
    ax.text(0.02, 0.98, textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(output_path,file_name+"png"), dpi=150)
    plt.close()
