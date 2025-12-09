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


from src_new.visualisation.utils_colormap import CMAP
from src_new.IR_to_SAR.data_preparation.distribution_data_visualisation import detect_sar_quadrants
from src_new.IR_to_SAR.data_preparation.distribution_data_visualisation import plot_ir, plot_sar,plot_ir_hist,plot_sar_hist



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

def z_score(tensor, eps = 1e-10):
    mean_value = np.mean(tensor)
    std_value = np.std(tensor)

    normalized_tensor = (tensor - mean_value)/(std_value + eps)

    return normalized_tensor, mean_value, std_value




# ============================================================
# ======================== split 3 sets ======================
# ============================================================
from collections import Counter
def train_val_test_split(
    ir_array,
    sar_array,
    mask_sar,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    n_bins=3,
    augmentation = True
):
    """
    Split IR→SAR dataset while preserving:
        - Brightness temperature distribution
        - Wind speed distribution
        - Quadrant spatial distribution

    Handles NaNs by temporary replacement with -1000 for stratification,
    but returns original tensors WITH true NaNs.
    """

    assert abs(train_size + val_size + test_size - 1) < 1e-6

    N = len(ir_array)

    # --- Create temporary clean versions (replace NaN by -1000) ---
    ir_clean = np.nan_to_num(ir_array, nan=-1000)
    sar_clean = np.nan_to_num(sar_array, nan=-1000)

    # --- 1️⃣ Compute median brightness & wind ---
    ir_median = np.median(ir_clean, axis=(1,2))
    sar_median = np.median(sar_clean, axis=(1,2))

    # --- 2️⃣ Encode into bins ---
    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    temp_bin = kb.fit_transform(ir_median.reshape(-1,1)).astype(int).flatten()
    wind_bin = kb.fit_transform(sar_median.reshape(-1,1)).astype(int).flatten()

    # --- 3️⃣ Quadrant detection using REAL NaN data ---
    quadrant_flags = []
    for i in range(N):
        q = detect_sar_quadrants(sar_array[i])  # Use original SAR with NaNs
        quadrant_flags.append("".join(map(str, q)))
    quadrant_flags = np.array(quadrant_flags)

    # --- 4️⃣ Create combined stratification label ---
    stratify_label = temp_bin.astype(str) + "_" + wind_bin.astype(str) + "_" + quadrant_flags

    # --- 5️⃣ Handle rare classes (<3 samples) ---
    counts = Counter(stratify_label)
    rare_classes = {k for k, v in counts.items() if v < 3}
    print(f"📉 {len(rare_classes)} rare stratification classes reassigned to 'OTHERS'")

    stratify_label_cleaned = np.array([
        lbl if lbl not in rare_classes else "OTHERS"
        for lbl in stratify_label
    ])

    # --- 6️⃣ Split train + temp (val+test) ---
    X = np.arange(N)
    X_train, X_temp, _, strat_temp = train_test_split(
        X,
        stratify_label_cleaned,
        stratify=stratify_label_cleaned,
        test_size=(1 - train_size),
        random_state=42
    )

    # --- 7️⃣ Split val/test ---
    val_rel_size = val_size / (val_size + test_size)

    # Re-check class frequencies in strat_temp
    counts_temp = Counter(strat_temp)
    rare_temp = {k for k, v in counts_temp.items() if v < 2}

    if rare_temp:
        print(f"🔁 Merging {len(rare_temp)} rare classes before val/test split...")
        strat_temp_cleaned = np.array([
            lbl if lbl not in rare_temp else "OTHERS"
            for lbl in strat_temp
        ])
    else:
        strat_temp_cleaned = strat_temp

    X_val, X_test = train_test_split(
        X_temp,
        stratify=strat_temp_cleaned,
        test_size=(1 - val_rel_size),
        random_state=42
    )
    print("🎯 Final Split Sizes:", len(X_train), len(X_val), len(X_test))

    # --- 8️⃣ Return ORIGINAL arrays (with true NaNs preserved) ---

    X_train = np.array([int(i) for i in X_train])
    X_val   = np.array([int(i) for i in X_val])
    X_test  = np.array([int(i) for i in X_test])

   
    ir_array = np.array(ir_array, dtype=float)
    sar_array = np.array(sar_array, dtype=float)
    ir_train,sar_train = ir_array[X_train], sar_array[X_train]

    #augmentation data for the train set (rottaion 90° and 180°)
    if augmentation : 
        ir_train, sar_train = data_augmentation(ir_train,sar_train)
        



    return {
        "train": (ir_train, sar_train),
        "val":   (ir_array[X_val],   sar_array[X_val]),
        "test":  (ir_array[X_test],  sar_array[X_test]),
        "mask_sar": mask_sar[X_val]
    }


def train_val_test_split_random(
    X_array,
    sar_array,
    mask_sar,
    cyclone_ids,
    sar_times,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    augmentation=True
):
    """
    Randomly split the dataset into train, validation, and test sets.
    Now also splits physical features X_features.
    """

    N = len(X_array)

    # --- Generate all indices ---
    all_indices = np.arange(N)

    # --- First split: Train vs Temp (Val+Test) ---
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=(1 - train_size),
        random_state=0,
        shuffle=True,
    )

    # --- Second split: Temp → Val & Test ---
    val_rel_size = val_size / (val_size + test_size)

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_rel_size),
        random_state=0,
        shuffle=True,
    )

    print("🎯 Split sizes — Train:", len(train_indices), 
          "| Val:", len(val_indices), 
          "| Test:", len(test_indices))

    # --- Convert to numpy ---
    X_array = np.array(X_array, dtype=float)
    sar_array = np.array(sar_array, dtype=float)
   

    # --- TRAIN ---
    ir_train  = X_array[train_indices]
    sar_train = sar_array[train_indices]

    mask_train = mask_sar[train_indices]
    cyclone_ids_train = cyclone_ids[train_indices]
    sar_time_train    = sar_times[train_indices]

    # --- Data augmentation only for train (images + SAR + mask) ---
    if augmentation:
        ir_train, sar_train, mask_train, cyclone_ids_train, sar_time_train = data_augmentation(
            ir_train,
            sar_train,
            mask_sar[train_indices],
            cyclone_ids_train,
            sar_time_train
        )
        
    # --- VAL ---
    ir_val   = X_array[val_indices]
    sar_val  = sar_array[val_indices]
    mask_val = mask_sar[val_indices]
    cyclone_ids_val = cyclone_ids[val_indices]
    sar_time_val    = sar_times[val_indices]

    # --- TEST ---
    ir_test   = X_array[test_indices]
    sar_test  = sar_array[test_indices]
    mask_test = mask_sar[test_indices]
    cyclone_ids_test = cyclone_ids[test_indices]
    sar_time_test    = sar_times[test_indices]

    # --- RETURN DICTIONARY ---
    return {
        # Images + SAR
        "train": (ir_train, sar_train),
        "val":   (ir_val, sar_val),
        "test":  (ir_test, sar_test),

        # Masks
        "mask_sar_train": mask_train,
        "mask_sar_val":   mask_val,
        "mask_sar_test":  mask_test,

        # Indices
        "val_index":   val_indices,
        "train_index": train_indices,
        "test_index":  test_indices,

        # Metadata
        "cyclone_id_train": cyclone_ids_train,
        "cyclone_id_val":   cyclone_ids_val,
        "cyclone_id_test":  cyclone_ids_test,

        "sar_time_train": sar_time_train,
        "sar_time_val":   sar_time_val,
        "sar_time_test":  sar_time_test,
    }

def crop_ir_to_sar(ir):
          
    H, W = ir.shape
    target = 301  # as  SAR data
    start_h = (H - target) // 2
    start_w = (W - target) // 2

    return ir[start_h:start_h+target, start_w:start_w+target]

def create_coloc_pkl(output_folder="/scale/user/mtannaou/alternance/src_new/IR_to_SAR/data_sar_ir_pkl", tcprimed = True):
    import pandas as pd
    import numpy as np
    import xarray as xr
    import os
    import pickle as pkl

    df = pd.read_csv("/scale/user/mtannaou/alternance/excels/SARGEO_SAR_v4_eyecenter.csv")
    SARGEO_PATH = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v00r00/cyclobs"
    SARAEQD_PATH = "/scale/user/mtannaou/alternance/donnees_sar_aeqd_eye"

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
        "owiwindspeed": [],    }     
        


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

    # Save to file
    output_path = os.path.join(output_folder, "ir_sar_pairs_eye_center_tcprimed.pkl")
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


def remove_sar_nan(X_batch, sar_batch, radius_km=100, km_per_pixel=2, threshold=0.5):
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

    for i in range(N):
        total = mask.sum()
        nan_pixels = np.isnan(sar_batch[i][mask]).sum()
        nan_ratio = nan_pixels / total

        if nan_ratio < threshold:
            X_filtered.append(X_batch[i])
            sar_filtered.append(sar_batch[i])
            kept_indices.append(i)

    return np.array(X_filtered), np.array(sar_filtered)





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


def realistic_cyclone_augmentation(ir, sar):
    """
    ir : numpy array (H,W) or (1,H,W)
    sar : numpy array (H,W)
    """
    # 1️⃣ Convertir numpy → Tensor (C,H,W)
    if isinstance(ir, np.ndarray):
        ir = torch.tensor(ir, dtype=torch.float32)
    if isinstance(sar, np.ndarray):
        sar = torch.tensor(sar, dtype=torch.float32)

    if ir.ndim == 2:  # (H,W) → (1,H,W)
        ir = ir.unsqueeze(0)
    if sar.ndim == 2:
        sar = sar.unsqueeze(0)

    transforms = T.Compose([
        T.RandomResizedCrop(256, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ])

    # 2️⃣ Appliquer les mêmes transformations aux deux images
    seed = np.random.randint(0, 100000)

    torch.manual_seed(seed)
    ir_aug = transforms(ir)

    torch.manual_seed(seed)
    sar_aug = transforms(sar)

    # 3️⃣ Retour numpy (H,W)
    ir_aug = ir_aug.squeeze(0).numpy()
    sar_aug = sar_aug.squeeze(0).numpy()

    return ir_aug, sar_aug


def data_augmentation(ir_tensor, sar_tensor, mask_tensor, cyc_ids, sar_times):
    
    ir_aug, sar_aug, mask_aug, cycs_aug, sar_times_aug = [], [], [], [], []

    for ir, sar, mask, cyc, sart in zip(ir_tensor, sar_tensor, mask_tensor, cyc_ids, sar_times):

        # original
        ir_aug.append(ir)
        sar_aug.append(sar)
        mask_aug.append(mask)
        cycs_aug.append(cyc)
        sar_times_aug.append(sart)

        # rotations
        for angle in [180, 90]:
            ir_r, sar_r, mask_r = augmentation_sar_safe(ir, sar, mask, angle=angle)
            ir_aug.append(ir_r)
            sar_aug.append(sar_r)
            mask_aug.append(mask_r)
            cycs_aug.append(cyc)
            sar_times_aug.append(sart)

    return (
        np.array(ir_aug),
        np.array(sar_aug),
        np.array(mask_aug),
        np.array(cycs_aug),
        np.array(sar_times_aug)
    )



def compute_global_distributions(model, dataloader, device,mean_val_sar, std_val_sar, save_dir, eps= 1e-10):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for ir, sar in dataloader:
            ir = ir.to(device)
            sar = sar.to(device)

            pred = model(ir, timestep=0).sample

            # Flatten valid pixels
            mask = np.isfinite(sar.cpu().numpy())
            sar = sar * (std_val_sar + eps) + mean_val_sar
            pred = pred * (std_val_sar + eps) + mean_val_sar
            sar_valid = sar[mask].cpu().numpy()
            pred_valid = pred[mask].cpu().numpy()

            all_true.append(sar_valid)
            all_pred.append(pred_valid)

    # Concatenate all batches
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    

    # Compute error
    all_errors = (all_pred - all_true) ** 2

    print(f"Collected {len(all_true)} valid SAR pixels for distribution analysis.")

    # ---- Plot global distributions ----
    #without counting the zero value
    plt.figure(figsize=(8,5))
    plt.hist(all_true, bins=50, alpha=0.6, density=False, label="Real SAR", color='blue')
    plt.hist(all_pred[all_pred > 5], bins=50, alpha=0.6, density=False, label="Predicted SAR", color='green')
    # plt.hist(all_errors[all_errors > 5], bins=50, alpha=0.6, density=False, label="Squared Error", color='red')
    plt.xlim(5, 150)

    plt.title("Global Distribution — Validation Dataset")
    plt.xlabel("Wind speed (knots) or error")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "wind_speed_distributions.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Saved global distribution plot: {save_path}")



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