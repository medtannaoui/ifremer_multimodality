"""
Data preprocessing utilities for IR → SAR learning.

This module provides:
    - Loading IR/SAR tensors from pkl
    - Resampling IR to SAR spatial resolution
    - SAR missing-value handling (mask or fill)
    - Adding channel dimension (N → N,1,H,W)
    - Normalizing tensors


"""
import warnings
warnings.filterwarnings("ignore")


import os
import numpy as np
import random
import xarray as xr
import pandas as pd
import pickle as pkl
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from src.visualisation.utils_colormap import CMAP
from src.IR_to_SAR.data_preparation.distribution_data_visualisation import detect_sar_quadrants
from src.IR_to_SAR.data_preparation.distribution_data_visualisation import plot_ir, plot_sar,plot_ir_hist,plot_sar_hist



# ============================================================
# ============== IR RESAMPLING TO SAR RESOLUTION =============
# ============================================================

def resize_ir_to_sar(
    ir_tensor: np.ndarray,
    max_radius_km: int = 300,
    sar_resolution_km: float = 0.5,
    plot: bool = False
):
    """
    Resamples IR image to SAR AEQD grid resolution.

    Inputs:
        ir_tensor          : (H, W) original IR image
        max_radius_km      : radius in km to crop (final image spans [-R,+R])
        sar_resolution_km  : desired SAR resolution (ex: 0.5 km)
        plot               : show before/after comparison

    Returns:
        ir_resized    : (H_sar, W_sar) IR resized to match SAR resolution
    """

    cmap = CMAP.cira_ir()

    # Center and crop the IR image to match SAR diameter (600 km)
    center_idx = np.array(ir_tensor).shape[0] // 2
    half_size  = max_radius_km // 2

    ir_crop = np.array(ir_tensor)[
        center_idx - half_size : center_idx + half_size,
        center_idx - half_size : center_idx + half_size
    ]  # shape = (300,300)

    # Compute scaling factor: original ~ 2 km/pixel → target sar_resolution (km/pixel)
    zoom_factor = 2.0 / sar_resolution_km
    ir_resized = zoom(ir_crop, zoom=zoom_factor, order=1)   # bilinear interpolation

    # Optional visualization
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Original crop
        ax[0].set_title("Original IR (cropped)")
        p0 = ax[0].imshow(ir_crop, cmap=cmap, origin="lower")
        plt.colorbar(p0, ax=ax[0])

        # Resized IR
        ax[1].set_title(f"IR resampled: {ir_resized.shape[0]}×{ir_resized.shape[1]} (resolution={sar_resolution_km} km)")
        p1 = ax[1].imshow(ir_resized, cmap=cmap, origin="lower")
        plt.colorbar(p1, ax=ax[1])

        plt.tight_layout()
        plt.show()

    return ir_resized


# ============================================================
# =================== SAR MISSING VALUE HANDLING =============
# ============================================================

def process_sar_missing_values(
    sar_tensor: np.ndarray,
    mode: str = "mask",
    fill_value: float = -1.0
):
    """
    Handles missing values (NaN) in SAR tensor.

    Inputs:
        sar_tensor : (H, W)
        mode       : "mask" → return tensor + mask
                      "fill" → fill NaN with a constant
        fill_value : value used if mode="fill"

    Returns:
        If mode="mask": (sar_tensor, sar_mask)
        If mode="fill": (sar_filled, sar_mask)

        sar_mask is boolean array of shape (H, W)
    """

    sar_mask = ~np.isnan(sar_tensor)

    if mode == "mask":
        return sar_tensor, sar_mask

    elif mode == "fill":
        sar_filled = sar_tensor.copy()
        sar_filled[np.isnan(sar_filled)] = fill_value
        return sar_filled, sar_mask

    else:
        raise ValueError("mode must be 'mask' or 'fill'")


# ============================================================
# ======================== NORMALIZATION ======================
# ============================================================


def min_max(tensor, eps = 1e-10):
    """
    Min-Max normalization of a NumPy tensor (N, H, W),
    mapping values to [0, 1] while preserving NaN values.
    
    Args:
        tensor (numpy.ndarray): Input tensor of shape (N, H, W)
    
    Returns:
        normalized_tensor (numpy.ndarray)
        min_val (float)
        max_val (float)
    """
    min_val = np.nanmin(tensor)
    max_val = np.nanmax(tensor)

    

    normalized_tensor = (tensor - min_val) / (max_val - min_val + eps)

    return normalized_tensor, min_val, max_val

def z_score(tensor, eps = 1e-10,mean_value=None,std_value=None):
    if mean_value is None:
        mean_value = np.mean(tensor)
        std_value = np.std(tensor)

    normalized_tensor = (tensor - mean_value)/(std_value + eps)

    return normalized_tensor, mean_value, std_value
            

def annular_normalization(
            images,
            bin_size=1,
            mask=None,
            stats=None
        ):
            N, H, W = images.shape
            cx,cy = H//2, W//2
            y, x = np.indices((H, W))    # y : same line has the same value x : same column have the same value
            radius = np.sqrt((y - cy)**2 + (x - cx)**2)
            radial_bins = (radius // bin_size).astype(np.int32)
            n_bins = radial_bins.max() + 1
            if mask is None:
                mask = np.isfinite(images)
            
            if stats is None:
                mean = np.zeros(n_bins)
                std = np.ones(n_bins)
                for b in range(n_bins):
                    pixels = images[:, radial_bins == b][mask[:, radial_bins == b]]
                    if pixels.size > 0:
                        mean[b] = pixels.mean()   #np.median(pixels)
                        std[b] =  pixels.std()    #np.percentile(pixels,75)-np.percentile(pixels,25) 
            else:
                mean = stats["mean"]
                std = stats["std"]
            images_norm = images.copy()
            for b in range(n_bins):
                m, s = mean[b], std[b]
                images_norm[:, radial_bins == b] = (
                    images_norm[:, radial_bins == b] - m
                ) / s
            stats = {
                "mean": mean,
                "std": std
            }
            return images_norm, stats

def annular_denormalization(
            images_norm,
            stats,
            bin_size=1
        ):
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


def subtract_radial_mean(
    images, 
    bin_size=1, 
    use_median=False, 
    mask=None,
    radial_profil = None
):
   
    N, H, W = images.shape
    cy, cx = H // 2, W // 2

    y, x = np.indices((H, W))
    radius = np.sqrt((y - cy)**2 + (x - cx)**2)
    radial_bins = (radius // bin_size).astype(np.int32)

    n_bins = radial_bins.max() + 1

    if mask is None:
        mask = np.isfinite(images)
    radial_profile = radial_profil
    if radial_profil is None:
        radial_profile = np.zeros(n_bins)
        for b in range(n_bins):
            pixels = images[:, radial_bins == b][mask[:, radial_bins == b]]
            if pixels.size > 0:
                if use_median:
                    radial_profile[b] = np.median(pixels)
                else:
                    radial_profile[b] = np.mean(pixels)
    
    images_anom = images.copy()
    for b in range(n_bins):
        images_anom[:, radial_bins == b] -= radial_profile[b]

    return images_anom, radial_profile



# ============================================================
# ======================== split 3 sets ======================
# ============================================================
from collections import Counter



import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


def train_val_test_split(
    X_array,
    sar_array,
    mask_sar,
    infos,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    augmentation=True,
    target_dir=None,
    n_bins=3,
):
   

    # assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    N = len(X_array)
    all_indices = np.arange(N)

    # --------------------------------------------------
    # 1) Extract stratification variables
    # --------------------------------------------------
    analysis_vmax = np.array([d["analysis_vmax"] for d in infos])
    analysis_rmax = np.array([d["analysis_rmax"] for d in infos])
    vmax = np.array([d["vmax"] for d in infos])

    delta_vmax = vmax - analysis_vmax

    # --------------------------------------------------
    # 2) Quantile binning (robust to imbalance)
    # --------------------------------------------------
    def quantile_bins(x, n_bins):
        q = np.nanquantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
        return np.digitize(x, q)

    vmax_bin  = quantile_bins(analysis_vmax, n_bins)
    rmax_bin  = quantile_bins(analysis_rmax, n_bins)
    # delta_bin = quantile_bins(delta_vmax, n_bins)

    # --------------------------------------------------
    # 3) Combine bins → single stratification key
    # --------------------------------------------------
    stratify_key = (
        vmax_bin.astype(str) + "_" +
        rmax_bin.astype(str) + "_" 
        # delta_bin.astype(str)
    )

    # --------------------------------------------------
    # 4) Remove rare classes (sklearn requirement)
    # --------------------------------------------------
    counts = Counter(stratify_key)
    valid_mask = np.array([counts[k] >= 2 for k in stratify_key])

    all_indices = all_indices[valid_mask]
    stratify_key = stratify_key[valid_mask]

    print(f"📊 Stratification classes kept: {len(set(stratify_key))}")

    # --------------------------------------------------
    # 5) Train vs Temp split
    # --------------------------------------------------
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=(1 - train_size),
        random_state=0,
        shuffle=True,
        stratify=stratify_key
    )

    # --------------------------------------------------
    # 6) Val vs Test split
    # --------------------------------------------------
    stratify_temp = stratify_key[np.isin(all_indices, temp_idx)]
    val_rel_size = val_size / (val_size + test_size)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_rel_size),
        random_state=0,
        shuffle=True,
        stratify=stratify_temp
    )

    print(
        "🎯 Stratified Split — "
        f"Train: {len(train_idx)} | "
        f"Val: {len(val_idx)} | "
        f"Test: {len(test_idx)}"
    )

    # --------------------------------------------------
    # 7) Convert arrays
    # --------------------------------------------------
    X_array = np.asarray(X_array, dtype=float)
    sar_array = np.asarray(sar_array, dtype=float)
    mask_sar = np.asarray(mask_sar)

    # --------------------------------------------------
    # 8) Build datasets
    # --------------------------------------------------
    ir_train   = X_array[train_idx]
    sar_train  = sar_array[train_idx]
    
    mask_train = mask_sar[train_idx]
    infos_train = [infos[i] for i in train_idx]

    val_idx = np.array(val_idx)
    analysis_rmax_val = np.array([infos[i]["analysis_rmax"] for i in val_idx])
    valid_mask = analysis_rmax_val < 100000
    val_idx_filtered = val_idx[valid_mask]
    ir_val   = X_array[val_idx_filtered]
    sar_val  = sar_array[val_idx_filtered]
    mask_val = mask_sar[val_idx_filtered]
    infos_val = [infos[i] for i in val_idx_filtered]


    test_idx = np.array(test_idx)
    analysis_rmax_test = np.array([infos[i]["analysis_rmax"] for i in test_idx])
    test_mask = analysis_rmax_test < 100000
    test_idx_filtered = test_idx[test_mask]
    ir_test   = X_array[test_idx_filtered]
    sar_test  = sar_array[test_idx_filtered]
    mask_test = mask_sar[test_idx_filtered]
    infos_test = [infos[i] for i in test_idx_filtered]

    # --------------------------------------------------
    # 9) Data augmentation (TRAIN ONLY)
    # --------------------------------------------------
    if augmentation:
        ir_train, sar_train, mask_train, infos_train = data_augmentation(
            ir_train, sar_train, mask_train, infos_train
        )

    plot_metric_scatter(
        true_values=[d["vmax"] for d in infos_train],
        pred_values=[d["analysis_vmax"] for d in infos_train],
        output_path=target_dir,
        file_name="analysis_vmax_and_vmax_comparaison_train",
        title="analysis vmax and vmax comparaison in the train set",
        xlabel="vmax",
        ylabel="analysis_vmax"
    )
    plot_metric_scatter(
        true_values=[d["vmax"] for d in infos_val],
        pred_values=[d["analysis_vmax"] for d in infos_val],
        output_path=target_dir,
        file_name="analysis_vmax_and_vmax_comparaison_val",
        title="analysis vmax and vmax comparaison in the val set",
        xlabel="vmax",
        ylabel="analysis_vmax"
    )
    plot_rmax_distribution(
        infos_train=infos_train,
        infos_val=infos_val,
        output_path=target_dir,
        file_name="analysis_rmax_distribution_train_vs_val.png"
    )

    # --------------------------------------------------
    # 10) Return dict (compatible with your pipeline)
    # --------------------------------------------------
    return {
        "train": (ir_train, sar_train),
        "val":   (ir_val, sar_val),
        "test":  (ir_test, sar_test),

        "mask_sar_train": mask_train,
        "mask_sar_val":   mask_val,
        "mask_sar_test":  mask_test,

        "train_index": train_idx,
        "val_index":   val_idx,
        "test_index":  test_idx,

        "infos_train": infos_train,
        "infos_val":   infos_val,
        "infos_test":  infos_test,
        
    }



def plot_rmax_distribution(infos_train, infos_val, output_path, file_name):
    import matplotlib.pyplot as plt
    import numpy as np

    rmax_train = np.array([d["analysis_rmax"] for d in infos_train if d["analysis_rmax"] is not None])
    rmax_val   = np.array([d["analysis_rmax"] for d in infos_val if d["analysis_rmax"] is not None])

    # Convert to km if needed
    rmax_train = rmax_train / 1000.0
    rmax_val   = rmax_val / 1000.0

    plt.figure(figsize=(7, 5))

    plt.hist(
        rmax_train, bins=30, alpha=0.6,
        label="Train", color="tab:blue", density=False
    )
    plt.hist(
        rmax_val, bins=30, alpha=0.6,
        label="Validation", color="tab:orange", density=False
    )

    plt.xlabel("Analysis Rmax (km)")
    plt.ylabel("Count")
    plt.title("Distribution of Analysis Rmax (Train vs Validation)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}/{file_name}", dpi=150)
    plt.close()

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
    """
    Generic scatter plot comparing true vs predicted metrics.
    """

    true_values = np.array(true_values, dtype=float)
    pred_values = np.array(pred_values, dtype=float)

    # Basic stats
    mean_true = np.nanmean(true_values)
    mean_pred = np.nanmean(pred_values)
    max_true  = np.nanmax(true_values)
    max_pred  = np.nanmax(pred_values)

    # Create figure
    plt.figure(figsize=(7, 7))
    plt.scatter(true_values, pred_values, alpha=0.5, color="#1f77b4", edgecolors="none")

    # Identity line
    min_v = min(true_values.min(), pred_values.min())
    max_v = max(true_values.max(), pred_values.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2)

    # Labels & title
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.5)

    # Save
    
    plt.savefig(os.path.join(output_path,file_name+"png"), dpi=150)
    plt.close()

def crop_ir_to_sar(ir):
          
    H, W = ir.shape
    target = 301  # as  SAR data
    start_h = (H - target) // 2
    start_w = (W - target) // 2

    return ir[start_h:start_h+target, start_w:start_w+target]

def create_coloc_pkl(output_folder="/scale/user/mtannaou/alternance/src/IR_to_SAR/data_sar_ir_pkl", tcprimed = False):
    import pandas as pd
    import numpy as np
    import xarray as xr
    import os
    import pickle as pkl

    df = pd.read_csv("/scale/user/mtannaou/alternance/excels/SARGEO_SAR_v5.csv")
    SARGEO_PATH = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v00r00/cyclobs"
    SARAEQD_PATH = "/scale/user/mtannaou/alternance/donnees_sar_aeqd_v3"

    CANAL = "IRWIN"

    data_pkl = {
        "cyclone_id": [],
        "sar_time": [],
        "env_time": [],
        "irwin": [],               
        "owiwindspeed": [],         
        "shear_magnitude": [],
        "shear_direction": [],
        "cyclone_phase_space_symmetry": [],
        "cyclone_phase_space_depth": [],
        "u_wind_mean": [],
        "v_wind_mean": [],
        "t_wind": [],
        "vorticity": []
    } if tcprimed else {
        "cyclone_id": [],
        "sar_time": [],
        "irwin": [],               
        "owiwindspeed": [], 
        "analysis_rmax":[],
        "analysis_vmax":[],
        "vmax":[],
        "analysis_center_quality_flag":[]  }     
        


    for i, row in df[df["canal"] == CANAL].iterrows():
        cyclone = row["cyclone"]

        if tcprimed : 
            if cyclone not in os.listdir(SARAEQD_PATH):
                continue
        else :
            for cyc in os.listdir(SARAEQD_PATH):
                if cyc[:8] == cyclone:
                    cyclone = cyc[:8]
                    break
        

        nc_sargeo_path = os.path.join(SARGEO_PATH, cyclone, CANAL, row["fichier"])
        nc_aeqd_path = row["sar_xy"]

        ds_aeqd = xr.open_dataset(nc_aeqd_path)
        ds_sargeo = xr.open_dataset(nc_sargeo_path)

        # Load data
        irwins = []
        for rel in [-4,-3,-2,-1,0,1,2,3,4]:
            irwin = crop_ir_to_sar(ds_sargeo["IRWIN"].sel(t_rel=rel).values)
            irwins.append(irwin)
        wind = ds_aeqd["owiWindSpeed"].values

        timestamp_str = os.path.basename(nc_aeqd_path).split("-")[4]
        sar_time = pd.to_datetime(timestamp_str, format="%Y%m%dT%H%M%S")
        # Find associated environmental file
        if tcprimed : 
            env_file = next((f for f in os.listdir(os.path.join(SARAEQD_PATH, cyclone)) if "TCPRIMED" in f), None)
            if not env_file:
                continue

            with xr.open_dataset(os.path.join(SARAEQD_PATH, cyclone, env_file), group="diagnostics") as ds_env:
                

                env_times = ds_env["time"].values
                idx = np.abs(env_times - np.datetime64(sar_time)).argmin()

                # Append values
                if tcprimed : 
                    data_pkl["cyclone_id"].append(cyclone)
                    data_pkl["sar_time"].append(sar_time)
                    data_pkl["env_time"].append(env_times[idx])
                    data_pkl["irwin"].append(irwins)
                    data_pkl["owiwindspeed"].append(wind)
                    data_pkl["shear_magnitude"].append(ds_env["shear_magnitude"].values[idx])
                    data_pkl["shear_direction"].append(ds_env["shear_direction"].values[idx])
                    data_pkl["cyclone_phase_space_symmetry"].append(ds_env["cyclone_phase_space_symmetry"].values[idx])
                    data_pkl["cyclone_phase_space_depth"].append(ds_env["cyclone_phase_space_depth"].values[idx])
                    data_pkl["u_wind_mean"].append(ds_env["u_wind_mean"].values[idx])
                    data_pkl["v_wind_mean"].append(ds_env["v_wind_mean"].values[idx])
                    data_pkl["t_wind"].append(ds_env["t_wind"].values[idx])
                    data_pkl["vorticity"].append(ds_env["vorticity"].values[idx])
        else :
                data_pkl["cyclone_id"].append(cyclone)
                data_pkl["sar_time"].append(sar_time)
                data_pkl["irwin"].append(irwins)
                data_pkl["owiwindspeed"].append(wind)
                data_pkl["analysis_center_quality_flag"].append(row["analysis_center_quality_flag"])
                data_pkl["vmax"].append(row["vmax"])
                data_pkl["analysis_vmax"].append(row["analysis_vmax"])
                data_pkl["analysis_rmax"].append(row["analysis_rmax"])
    # Save to file
    output_path = os.path.join(output_folder, "ir_sar_sargeo_with_cyclobs_infos.pkl")
    with open(output_path, "wb") as f:
        pkl.dump(data_pkl, f)

    print(f"🎯 PKL file created: {output_path}")




# recentring the sar data around their barycenter

def recenter_sar_around_barycenter(sars):
    """
    Recentre l'image SAR par translation (shift) sans interpolation.
    - sar : tableau 2D (H, W) contenant des NaN pour les valeurs invalides.
    Retourne :
    - sar_shifted : image translatée, même shape
    - barycenter (x, y) en pixels
    - shift (dx, dy) appliqué
    """
    sars_recenter = []
    # get the valid pixel valeus
    for sar in sars:
        mask = ~np.isnan(sar)
        if not np.any(mask):
            sars_recenter.append((sar.copy(), 0, 0))
            continue

        ys, xs = np.where(mask)

        x_center = xs.mean()
        y_center = ys.mean()
        barycenter = (x_center, y_center)

        H, W = sar.shape
        target_x = W // 2
        target_y = H // 2

        dx = int(target_x - x_center)
        dy = int(target_y - y_center)

        sar_shifted = np.roll(sar, shift=(dy, dx), axis=(0, 1))
        sars_recenter.append((sar_shifted,dx,dy))

    return sars_recenter


def get_mask_of_nan_values(tensor, invalid_values=None):
    mask = (~torch.isnan(tensor)) & (~torch.isinf(tensor))
    
    # Gestion des valeurs spécifiques invalides (ex: -999, 0)
    if invalid_values is not None:
        for val in invalid_values:
            mask &= tensor != val
    
    return mask.float()


def get_nan_coverage(sar_batch, radius=100, km_per_pixel=2):
    results = []
    radius_pix = int(radius / km_per_pixel)

    for idx, sar in enumerate(sar_batch):
        H, W = sar.shape
        x0, y0 = W // 2, H // 2  # centre de l’image

        y, x = np.ogrid[:H, :W]
        mask = (x - x0) ** 2 + (y - y0) ** 2 <= radius_pix ** 2

        total_pixels = mask.sum()
        nan_pixels = np.isnan(sar[mask]).sum()

        nan_ratio = nan_pixels / total_pixels
        results.append((idx, nan_ratio))

        print(f"Image {idx}: {nan_ratio*100:.2f}% de NaN dans r ≤ {radius} km")

    return results


def remove_sar_nan(X_batch, sar_batch, radius_km=100, km_per_pixel=2, threshold=0.5,infos=None):
    """
    X_batch : numpy array (N, C, H, W) or (N, 1, H, W)
    sar_batch : numpy array (N, H, W)
    Returns:
      - X_filtered (N_filtered, C, H, W)
      - sar_filtered (N_filtered, H, W)
      - kept_indices (list of indices kept)
    """

    # Convert tensors to numpy
    if isinstance(X_batch, torch.Tensor):
        X_batch = X_batch.cpu().numpy()
    if isinstance(sar_batch, torch.Tensor):
        sar_batch = sar_batch.cpu().numpy()

    N, C, H, W = X_batch.shape
    assert sar_batch.shape == (N, H, W), "Shapes do not match"

    # Create circular mask
    x0, y0 = W // 2, H // 2
    radius_pix = int(radius_km / km_per_pixel)

    y, x = np.ogrid[:H, :W]
    mask = (x - x0)**2 + (y - y0)**2 <= radius_pix**2

    X_filtered = []
    sar_filtered = []
    kept_indices = []
    infos_kept = []

    for i in range(N):
        total = mask.sum()
        nan_pixels = np.isnan(sar_batch[i][mask]).sum()
        nan_ratio = nan_pixels / total

        if nan_ratio < threshold:
            X_filtered.append(X_batch[i])
            sar_filtered.append(sar_batch[i])
            kept_indices.append(i)
            infos_kept.append(infos[i])

    return np.array(X_filtered), np.array(sar_filtered),np.array(infos_kept)





def augmentation_sar_safe(ir, sar, mask, angle=None, flip=None):
    
    ir_t = torch.tensor(ir, dtype=torch.float32).unsqueeze(0)
    sar_t = torch.tensor(sar, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    if angle is not None:
        ir_t = TF.rotate(ir_t, angle)
        sar_t = TF.rotate(sar_t, angle)
        mask_t = TF.rotate(mask_t, angle)

    if flip == "h":
        ir_t = TF.hflip(ir_t)
        sar_t = TF.hflip(sar_t)
        mask_t = TF.hflip(mask_t)

    elif flip == "v":
        ir_t = TF.vflip(ir_t)
        sar_t = TF.vflip(sar_t)
        mask_t = TF.vflip(mask_t)

    return (
        ir_t.squeeze(0).numpy(),
        sar_t.squeeze(0).numpy(),
        mask_t.squeeze(0).numpy(),
    )



def data_augmentation(ir_tensor, sar_tensor, mask_tensor, infos):
    
    ir_aug, sar_aug, mask_aug, infos_aug = [], [], [], []

    for ir, sar, mask, inf in zip(ir_tensor, sar_tensor, mask_tensor, infos) :
        # print("rmax_unities llllllll",np.nanmax(np.array(inf["analysis_vmax"])))
        if np.nanmax(np.array(inf["analysis_rmax"])) > 100000:     # delete ctorms with radisu of max wind speed less than 100km
            continue
        # original
        ir_aug.append(ir)
        sar_aug.append(sar)
        mask_aug.append(mask)
        infos_aug.append(inf)

        # rotations
        
        if np.nanmax(np.array(inf["analysis_vmax"])) > 50:   # for ctorms with max wind speed more than 50m/s
            for angle in [180,90,270]:
                ir_r, sar_r, mask_r = augmentation_sar_safe(ir, sar, mask, angle=angle)
                ir_aug.append(ir_r)
                sar_aug.append(sar_r)
                mask_aug.append(mask_r)
                infos_aug.append(inf)
        # print(np.nanmax(np.array(inf["analysis_rmax"])))
        if np.nanmax(np.array(inf["analysis_rmax"])) > 30000:   # for ctorms with radius of max wind speed more than 50km
            for flip in ["h","v"]:
                ir_r, sar_r, mask_r = augmentation_sar_safe(ir, sar, mask, flip=flip)
                ir_aug.append(ir_r)
                sar_aug.append(sar_r)
                mask_aug.append(mask_r)
                infos_aug.append(inf)

    return (
        np.array(ir_aug),
        np.array(sar_aug),
        np.array(mask_aug),
        np.array(infos_aug)
    )


def visualize_dataset_statistics(dictio, target_dir, mask=None):
    """
    Génère une figure 4x2 :
        1) Moyenne IR   (train / val)
        2) Moyenne SAR  (train / val)
        3) Histogrammes IR (train / val)
        4) Histogrammes SAR (train / val)
    """

    # --- Récupération des datasets ---
    train_ir = dictio["train"][0]          # (N,C,H,W)
    train_sar = dictio["train"][1]         # (N,H,W)
    val_ir   = dictio["val"][0]            # (N,C,H,W)
    val_sar  = dictio["val"][1]            # (N,H,W)

    # --- Récupération masques ---
    mask_train = dictio["mask_sar_train"]  # (N,H,W)
    mask_val   = dictio["mask_sar_val"]    # (N,H,W)

    # --- Moyennes ---
    mean_train_ir  = np.nanmean(train_ir[:,0], axis=0)
    mean_val_ir    = np.nanmean(val_ir[:,0], axis=0)

    mean_train_sar = np.nanmean(train_sar, axis=0)
    mean_val_sar   = np.nanmean(val_sar, axis=0)

    # --- Figure ---
    fig, axs = plt.subplots(4, 2, figsize=(18, 20))

    # -------------------------------
    # 1) IR MEAN IMAGES
    # -------------------------------
    plot_ir(mean_train_ir, ax=axs[0,0])
    axs[0,0].set_title("Train — Mean IR")

    plot_ir(mean_val_ir, ax=axs[0,1])
    axs[0,1].set_title("Validation — Mean IR")

    # -------------------------------
    # 2) SAR MEAN IMAGES
    # -------------------------------
    plot_sar(mean_train_sar, ax=axs[1,0])
    axs[1,0].set_title("Train — Mean SAR")

    plot_sar(mean_val_sar, ax=axs[1,1])
    axs[1,1].set_title("Validation — Mean SAR")

    # -------------------------------
    # 3) IR HISTOGRAMS (mask ignored)
    # -------------------------------
    ir_train_flat = train_ir[:,0].reshape(-1)
    ir_val_flat = val_ir[:,0].reshape(-1)

    ir_train_valid = ir_train_flat[np.isfinite(ir_train_flat)]
    ir_val_valid   = ir_val_flat[np.isfinite(ir_val_flat)]

    plot_ir_hist(ir_train_valid, ax=axs[2,0])
    axs[2,0].set_title("Train — IR Histogram (no NaN)")

    plot_ir_hist(ir_val_valid, ax=axs[2,1])
    axs[2,1].set_title("Validation — IR Histogram (no NaN)")

    # -------------------------------
    # 4) SAR HISTOGRAMS (mask applied)
    # -------------------------------
    sar_train_flat = train_sar.reshape(-1)
    sar_val_flat   = val_sar.reshape(-1)

    mask_train_flat = mask_train.reshape(-1)
    mask_val_flat   = mask_val.reshape(-1)

    # garder uniquement les pixels où mask == 1
    sar_train_valid = sar_train_flat[(mask_train_flat == 1) & np.isfinite(sar_train_flat)]
    sar_val_valid   = sar_val_flat[(mask_val_flat == 1) & np.isfinite(sar_val_flat)]

    plot_sar_hist(sar_train_valid, ax=axs[3,0])
    axs[3,0].set_title("Train — SAR Histogram (mask applied)")

    plot_sar_hist(sar_val_valid, ax=axs[3,1])
    axs[3,1].set_title("Validation — SAR Histogram (mask applied)")

    plt.tight_layout()

    # --- Save ---
    save_dir = os.path.join(target_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "dataset_overview.png")
    fig.savefig(save_path, dpi=200)

    plt.close(fig)
    plt.close("all")

    print(f"📊 Dataset visualization saved to: {save_path}")


    
if __name__ =="__main__":
    create_coloc_pkl()