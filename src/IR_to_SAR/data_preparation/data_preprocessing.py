import warnings
warnings.filterwarnings("ignore")

import os
import re
import copy
import json
from tqdm import tqdm

import numpy as np
import xarray as xr

from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from datetime import datetime, timedelta    
import torch
from torchvision.transforms import functional as TF
import torchvision.transforms as T
import pyresample
import pyproj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.IR_to_SAR.data_preparation.distribution_data_visualisation import *


# ======================== NORMALIZATION ======================
def min_max(tensor, eps = 1e-10):
    min_val = np.nanmin(tensor)
    max_val = np.nanmax(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val + eps)
    return normalized_tensor, min_val, max_val


def z_score(tensor, eps=1e-10, mean_value=None, std_value=None, mask=None):
    
    if mean_value is None:
        if mask is None:
            mean_value = np.nanmean(tensor)
            std_value = np.nanstd(tensor)
        else:
            valid = tensor[mask.astype(bool)]
            mean_value = valid.mean()
            std_value = valid.std()

    normalized_tensor = (tensor - mean_value) / (std_value + eps)

    if mask is not None:
        normalized_tensor = np.where(mask, normalized_tensor, 0)

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
                mean[b] = pixels.mean()   #np.median(pixels) #
                std[b] =  pixels.std()    #np.percentile(pixels,75)-np.percentile(pixels,25) #
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

##### AUgmentation #######################
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

def add_white_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    return img + noise

def add_salt_pepper_noise(img, amount=0.01, salt_vs_pepper=0.5):
    noisy = img.copy()
    num_pixels = int(amount * img.size)
    num_salt = int(num_pixels * salt_vs_pepper)
    coords = tuple(
        np.random.randint(0, i, num_salt)
        for i in img.shape
    )
    noisy[coords] = np.max(img)
    num_pepper = int(num_pixels * (1.0 - salt_vs_pepper))
    coords = tuple(
        np.random.randint(0, i, num_pepper)
        for i in img.shape
    )
    noisy[coords] = np.min(img)
    return noisy

def create_moment_sar(sar):
    assert sar.ndim == 3, "sar must be (N, H, W)"
    N, H, W = sar.shape
    y, x = np.indices((H, W))
    cy, cx = H // 2, W // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    moment = sar * r[None, :, :]
    return moment

def moment_to_sar(moment):
    assert moment.ndim == 3, "moment must be (N, H, W)"
    N, H, W = moment.shape
    y, x = np.indices((H, W))
    cy, cx = H // 2, W // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_safe = np.maximum(r, 1.0)
    sar = moment / r_safe[None, :, :]
    return sar

def data_augmentation(ir_tensor, sar_tensor, mask_tensor, infos):
    out_ir, out_sar, out_mask, out_infos = [], [], [], []
    for ir, sar, mask, inf in zip(ir_tensor, sar_tensor, mask_tensor, infos):
        inf0 = copy.deepcopy(inf)
        inf0["augmentation"] = 0
        out_ir.append(ir)
        out_sar.append(sar)
        out_mask.append(mask)
        out_infos.append(inf0)
        rmax = inf.get("analysis_rmax", np.nan)
        rmax = np.nanmax(rmax) if np.ndim(rmax) > 0 else float(rmax)
        vmax = inf.get("analysis_vmax", np.nan)
        vmax = np.nanmax(vmax) if np.ndim(vmax) > 0 else float(vmax)
        if np.isfinite(rmax) and (rmax > 0):
            for flip in ["h","v"]:
                ir_r, sar_r, mask_r = augmentation_sar_safe(ir, sar, mask, flip=flip)
                inf_aug = copy.deepcopy(inf)
                inf_aug["augmentation"] = 1
                inf_aug["aug_type"] = f"flip_{flip}"
                out_ir.append(ir_r)
                out_sar.append(sar_r)
                out_mask.append(mask_r)
                out_infos.append(inf_aug)

            for angle in [90,270,180]:
                ir_r, sar_r, mask_r = augmentation_sar_safe(ir, sar, mask, angle=angle)
                inf_aug = copy.deepcopy(inf)
                inf_aug["augmentation"] = 1
                inf_aug["aug_type"] = f"rot_{angle}"
                out_ir.append(ir_r)
                out_sar.append(sar_r)
                out_mask.append(mask_r)
                out_infos.append(inf_aug)

        if np.isfinite(rmax) and (rmax > 0):
            for sigma in [0.05]:
                ir_r = add_white_noise(ir, sigma)
                inf_aug = copy.deepcopy(inf)
                inf_aug["augmentation"] = 1
                inf_aug["aug_type"] = f"white_noise_{sigma}"
                out_ir.append(ir_r)
                out_sar.append(sar)   
                out_mask.append(mask)
                out_infos.append(inf_aug)
        if np.isfinite(rmax) and (vmax < 10000):
            for amount in [0.04]:
                ir_r = add_salt_pepper_noise(ir, amount)
                inf_aug = copy.deepcopy(inf)
                inf_aug["augmentation"] = 1
                inf_aug["aug_type"] = f"saltpepper_{amount}"
                out_ir.append(ir_r)
                out_sar.append(sar)      # unchanged
                out_mask.append(mask)
                out_infos.append(inf_aug)
    return (
        np.stack(out_ir, axis=0),
        np.stack(out_sar, axis=0),
        np.stack(out_mask, axis=0),
        out_infos
    )

def regrid_batch_by_resolution(x, in_resolution, out_resolution):
    scale = out_resolution / in_resolution  # ex: 4/2 = 2
    if np.isclose(scale, 1.0):
        return x
    if scale < 1:
        raise ValueError("This implementation handles downsampling only (out_resolution > in_resolution).")
    factor = int(np.round(scale))
    N, C, H, W = x.shape
    Hc = (H // factor) * factor
    Wc = (W // factor) * factor
    x = x[:, :, :Hc, :Wc]
    x = x.reshape(N, C, Hc // factor, factor, Wc // factor, factor)
    x = x.mean(axis=(3, 5))
    return x


def compute_residual_stats(train_loader, regression_model, device="cpu"):
    regression_model.eval()
    sum_vals  = 0.0
    sum_sq    = 0.0
    n_pixels  = 0
    with torch.no_grad():
        for x, sar, mask, infos in tqdm(train_loader, desc="Computing residual stats"):
            x    = x.to(device)
            sar  = sar.to(device)
            mask = mask.to(device)
            if sar.ndim == 3:
                sar  = sar.unsqueeze(1)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            t0 = torch.zeros(x.shape[0], device=device)
            mean_pred = regression_model(x, t0).sample   # (B, 1, H, W)
            residual = sar - mean_pred
            valid    = (mask > 0) & sar.isfinite()
            r_valid = residual[valid]
            sum_vals += r_valid.sum().item()
            sum_sq   += (r_valid ** 2).sum().item()
            n_pixels += r_valid.numel()
    mean = sum_vals / n_pixels
    var  = sum_sq   / n_pixels - mean ** 2
    std  = max(var, 1e-8) ** 0.5
    stats = {"mean": mean, "std": std, "n_pixels": n_pixels}
    print(f"Residual stats: mean={mean:.4f}, std={std:.4f} (n_pixels={n_pixels:,})")
    return stats


def save_residual_stats(stats: dict, path: str):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved residual stats to {path}")


def load_residual_stats(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def shift_ir_path(ir_path: str, idx: int, step_minutes: int = 30) -> str:
    m = re.search(r"(IR_)(\d{14})(\.nc)$", ir_path)
    if not m:
        raise ValueError(f"Format inattendu pour ir_path: {ir_path}")
    prefix, dt_str, suffix = m.group(1), m.group(2), m.group(3)
    dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    dt_shifted = dt + timedelta(minutes=idx * step_minutes)
    new_name = f"{prefix}{dt_shifted.strftime('%Y%m%d%H%M%S')}{suffix}"
    return ir_path[: m.start()] + new_name  # garde le même dossier


def build_storm_centered_dataset(
    ds,
    half_size=500,
    dxy=2.0,
    kind="nearest"
):
    ir = ds["brightness_temperature"].values
    ir_lon = ((ds["longitude"].values+180)%360)-180
    ir_lat = ds["latitude"].values
    center_lat = ds["storm_latitude"].values[0]
    center_lon = ((ds["storm_longitude"][0].values + 180)%360)-180
    nx = ny = int(2 * half_size / dxy) + 1
    x = np.linspace(-half_size, half_size, nx)
    y = np.linspace(half_size, -half_size, ny)  # nord en haut
    proj = pyproj.Proj(              #place le cyclone au centre (0,0), et je mesure tout en km autou
        proj="aeqd",
        lat_0=center_lat,
        lon_0=center_lon,
        ellps="WGS84",
        units="km"
    )
    longitude, latitude = pyresample.utils.check_and_wrap(ir_lon, ir_lat)
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    swath_ir = pyresample.SwathDefinition(
        lon2d,
        lat2d
    )
    area_def = pyresample.geometry.AreaDefinition(
        "storm",
        "storm centered grid",
        "storm",
        proj.srs,
        nx,
        ny,
        (x[0] - dxy/2, y[-1] - dxy/2, x[-1] + dxy/2, y[0] + dxy/2)
    )
    if kind == "nearest":
        ir_grid = pyresample.kd_tree.resample_nearest(
            swath_ir,
            ir,
            area_def,
            radius_of_influence=100000,
            fill_value=np.nan
        )
    else:
        ir_grid = pyresample.kd_tree.resample_gauss(
            swath_ir,
            ir,
            area_def,
            radius_of_influence=100000,
            sigmas=25000
        )
    ds["ir_aeqd"] = xr.DataArray(
            ir_grid,
            dims=("y", "x"),
            coords={
                "x": x,
                "y": y,
            },
            attrs={
                "units": ds["brightness_temperature"].attrs.get("units", ""),
                "description": "Brightness temperature projected onto storm-centered AEQD grid",
            },
        )
    ds["center_lat"] = center_lat
    ds["center_lon"] = center_lon
    return ds


def custom_collate(batch):
    """
    batch = [
        (x, sar, mask, infos_dict),
        ...
    ]
    """
    xs    = torch.stack([item[0] for item in batch])
    sars  = torch.stack([item[1] for item in batch])
    masks = torch.stack([item[2] for item in batch])

    # infos reste une liste de dictionnaires
    infos = [item[3] for item in batch]

    return xs, sars, masks, infos

import src.IR_to_SAR.data_preparation.regrid_era5.regrid_era5 as regrid_colocs
era5_path = "/scale/user/mtannaou/alternance/src/extract_cyclones_era5/era5_single_levels"
janvier, mars, mai, juillet, aout, octobre, decembre = (
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
    np.arange(1, 32, 1),
)
avril, juin, septembre, novembre = (
    np.arange(1, 31, 1),
    np.arange(1, 31, 1),
    np.arange(1, 31, 1),
    np.arange(1, 31, 1),
)


def add_era5_irar(path):
    base_path = os.path.basename(path)
    cyclone_id = str(base_path).split("_s20")[0].split("_")[-1]
    date = str(base_path).split("_s")[-1].split("_")[0]
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    hour = date[8:10]
    minute = date[10:12]
    year_path = os.path.join(era5_path, str(year))
    fevrier = np.arange(1, 29, 1) if int(year) % 4 != 0 else np.arange(1, 30, 1)
    months = [
        janvier, fevrier, mars, avril, mai, juin,
        juillet, aout, septembre, octobre, novembre, decembre,
    ]
    ndays = 0
    for i in range(int(month) - 1):
        ndays += len(months[i])
    ndays += int(day)
    ndays_str = "0" + str(ndays) if len(str(ndays)) < 3 else str(ndays)
    dayera5_path = os.path.join(year_path, ndays_str)
    nc_path = ""
    for nc_file in os.listdir(dayera5_path):
        if str(cyclone_id).lower() in nc_file:
            nc_path = os.path.join(dayera5_path, nc_file)
            break
    reg_era5 = regrid_colocs.regrid_files_era5(
        [nc_path],
        "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/"
        "regrid_era5/regridded_era5",
        resolution_km=2,
        grid_size_km=512,
        index_hour=int(hour) - 1 + int(minute) // 30,
    )[0]

    return reg_era5

def add_era5_anggrek(anggrek_csv, row_idx, irar):
    cyclone_id = anggrek_csv.iloc[row_idx]["sid"]
    date = anggrek_csv.iloc[row_idx]["date"]
    year = date[0:4]
    month = date[5:7]
    day = date[8:10]
    hour_i = date[11:13]
    minute = date[14:16]
    era5s = []
    reg_era5_all = []

    for delta in range(-5, 6):
        hour = int(hour_i) + delta
        day_tmp = int(day)

        if hour < 0:
            hour += 24
            day_tmp -= 1
        elif hour >= 24:
            hour -= 24
            day_tmp += 1

        year_path = os.path.join(era5_path, str(year))

        fevrier = np.arange(1, 29, 1) if int(year) % 4 != 0 else np.arange(1, 30, 1)
        months = [
            janvier, fevrier, mars, avril, mai, juin,
            juillet, aout, septembre, octobre, novembre, decembre,
        ]

        ndays = 0
        for i in range(int(month) - 1):
            ndays += len(months[i])
        ndays += day_tmp

        ndays_str = f"{ndays:03d}"
        dayera5_path = os.path.join(year_path, ndays_str)

        nc_path = ""
        for nc_file in os.listdir(dayera5_path):
            if cyclone_id in nc_file:
                nc_path = os.path.join(dayera5_path, nc_file)
                break

        reg_era5 = regrid_colocs.regrid_files_era5(
            [nc_path],
            "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/"
            "regrid_era5/regridded_era5",
            resolution_km=2,
            grid_size_km=1000 if not irar else 512,
            index_hour=hour - 1 + int(minute) // 30,
        )

        reg_era5_all.append(reg_era5)
    reg_era5_all = np.stack(reg_era5_all, axis=0)

    return np.stack(reg_era5_all, axis=0)