"""
Module de préparation des données pour l'entraînement IR -> SAR.

Ce module définit la classe `PrepareDataSet` qui :
    1. Charge le fichier CSV de correspondance entre les images IR (IRWIN) et SAR.
    2. Vérifie l'ouverture effective des fichiers NetCDF (SAR + IRWIN), et le cas
       échéant des données ERA5 associées.
    3. Construit les tenseurs d'entrée (X) et de sortie (sar) pour les ensembles
       train / val / test (et éventuellement le jeu de test "Anggrek").
    4. Applique le recadrage spatial, le downsampling, la conversion Kelvin -> Celsius,
       le masquage des NaN, l'augmentation de données et la normalisation (z-score
       ou annulaire).
    5. Sauvegarde les statistiques de normalisation ainsi que quelques graphiques
       de contrôle qualité.

NOTE IMPORTANTE : aucune logique métier n'a été modifiée par rapport au script
d'origine. Seuls ont été retirés les imports et variables qui n'étaient jamais
utilisés, et le code a été réorganisé/commenté pour en faciliter la lecture.
"""

import os
import re
from datetime import datetime, timedelta
from importlib import reload

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import pickle as pkl

import src.IR_to_SAR.data_preparation.data_preprocessing as dataprep
import src.IR_to_SAR.data_preparation.regrid_era5.regrid_era5 as regrid_colocs

# Recharge le module dataprep (utile en environnement interactif / notebook)
reload(dataprep)


# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# Répertoire racine contenant les fichiers ERA5 (un sous-dossier par année,
# puis par jour de l'année).
era5_path = "/scale/user/mtannaou/alternance/src/extract_cyclones_era5/era5_single_levels"

# Tableaux représentant les jours de chaque mois (utilisés pour calculer le
# numéro du jour dans l'année, nécessaire pour retrouver le bon sous-dossier
# ERA5). Les mois à 31 jours et à 30 jours sont définis séparément ; le mois
# de février est recalculé dynamiquement (année bissextile ou non) au moment
# de son utilisation plus bas dans le code.
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


def shift_ir_path(ir_path: str, idx: int, step_minutes: int = 30) -> str:
    """
    Décale le timestamp contenu dans le nom d'un fichier IR (format
    "IR_YYYYMMDDHHMMSS.nc") d'un certain nombre de pas de temps.

    Args:
        ir_path: chemin complet du fichier IR de référence.
        idx: nombre de pas de temps à décaler (peut être négatif).
        step_minutes: durée en minutes d'un pas de temps (30 min par défaut).

    Returns:
        Le chemin du fichier IR correspondant au timestamp décalé, dans le
        même dossier que le fichier d'origine.
    """
    m = re.search(r"(IR_)(\d{14})(\.nc)$", ir_path)
    if not m:
        raise ValueError(f"Format inattendu pour ir_path: {ir_path}")

    prefix, dt_str, suffix = m.group(1), m.group(2), m.group(3)
    dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    dt_shifted = dt + timedelta(minutes=idx * step_minutes)

    new_name = f"{prefix}{dt_shifted.strftime('%Y%m%d%H%M%S')}{suffix}"
    return ir_path[: m.start()] + new_name  # garde le même dossier


class PrepareDataSet:
    """
    Construit et normalise les jeux de données (train / val / test / Anggrek)
    utilisés pour entraîner les modèles IR -> SAR.
    """

    def __init__(
        self,
        pkl_file=None,
        input_channels=None,
        barycenter="no",
        size=128,
        norm="z_score",
        drop_nan_100=True,
        train_split=None,
        val_split=None,
        test_split=None,
        augmentation=False,
        target_dir=None,
        input_data="normal",
        output_data="sar",
        conditional_model=None,
        anggrek_test=False,
        log_wind=False,
        irwin_channels=1,
        regrid_ir=False,
        ir_smoothing=False,
        add_era5=False,
        cfg=None,
    ):
        # -------------------------------------------------------------
        # Sauvegarde des paramètres de configuration sur l'instance
        # -------------------------------------------------------------
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.augmentation = augmentation
        self.target_dir = target_dir
        self.input_data = input_data
        self.output_data = output_data
        self.log_wind = log_wind
        self.irwin_channels = irwin_channels
        self.regrid_ir = regrid_ir
        self.ir_smoothing = ir_smoothing
        self.add_era5 = add_era5
        self.cfg = cfg

        # -------------------------------------------------------------
        # 1) Chargement du CSV de correspondance IR / SAR
        # -------------------------------------------------------------
        print("🔹 Loading data from csv ...")

        irwin_train, irwin_val, irwin_test, irwin_anggrek = [], [], [], []
        sar_train, sar_val, sar_test = [], [], []
        infos_train, infos_val, infos_test, infos_anggrek = [], [], [], []
        era5_train, era5_val, era5_test, era5_anggrek = [], [], [], []
        if self.cfg.irar:
            data = pd.read_csv(
                "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/IRAR_L2_dataset.csv"
            )
            irar = pd.read_csv("/scale/user/mtannaou/alternance/mnt/csvs_finaux/IRAR.csv")


        else :
            if not conditional_model:
                if not self.regrid_ir:
                    data = pd.read_csv(
                        "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/"
                        "TCVA_matched_with_SARGEO_v3_split_by_year.csv"
                    )[: 50 if self.cfg.code_test else None]
                else:
                    data = pd.read_csv(
                        "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/"
                        "tcva_matched_sargeo_4km_resolution.csv"
                    )[:]
            else:
                data = pd.read_csv(
                    "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/"
                    "TCVA_matched_with_SARGEO_tcprimed.csv"
                )
                data = data[~data["tcprimed_env_path"].isna()]

            # Données du cyclone "Anggrek" (jeu de test additionnel)
            anggrek_csv = pd.read_csv(
                "/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/"
                "anggrek_coloc_sar_ir.csv"
            )
            print("data before filtering", len(data))

            # On ne garde que les lignes dont le split (train/val/test) est défini
            data = data[~data["split"].isna()]
            print("data after filtering", len(data))
            data = data.reset_index(drop=True)

            # -------------------------------------------------------------
            # 2) Construction des listes d'images / infos par split
            #    (uniquement si on n'est pas en mode "irar")
            # -------------------------------------------------------------
            # --- Normalisation des features de cisaillement (shear) si modèle conditionnel ---
            if conditional_model:
                mag_cols = [f"shear_magnitude_{i}" for i in range(1, 9)]
                dir_cols = [f"shear_direction_{i}" for i in range(1, 9)]
                shear_cols = dir_cols + mag_cols

                train_shear = data[data["split"] == "train"][shear_cols].to_numpy(dtype=np.float32)
                train_shear = np.nan_to_num(train_shear, nan=0.0)

                shear_mean = train_shear.mean(axis=0)  # (16,)
                shear_std = train_shear.std(axis=0)    # (16,)

            # Métadonnées conservées pour chaque échantillon
            keys = [
                "cyclone_name",
                "cyclone_id",
                "sar_time",
                "vmax",
                "analysis_vmax",
                "analysis_rmax",
                "analysis_center_quality_flag",
            ]  # corilis

            good_rows, bad_rows = [], []

            # ===========================================================
            # 2.a) Ensemble TRAIN
            # ===========================================================
            N = len(data[data["split"] == "train"])
            pbar = tqdm(total=N, desc="Checking train files", unit="row")
            for i, row in data[data["split"] == "train"].iterrows():
                try:
                    with xr.open_dataset(row["sargeo_path"]) as sargeo:
                        if "IRWIN" not in sargeo:
                            raise KeyError("Missing IRWIN")

                    with xr.open_dataset(row["sar_aeqd_path"]) as ds_aeqd:
                        if "owiWindSpeed" not in ds_aeqd:
                            raise KeyError("Missing owiWindSpeed")

                        # --- Récupération éventuelle des données ERA5 colocalisées ---
                        if self.add_era5:
                            sar_path = row["sar_aeqd_path"]
                            list_sar_path = [sar_path]
                            cyclone_id = row["cyclone_id"]
                            date = str(sar_path.split("/")[-1].split("-")[5])
                            year = date[0:4]
                            month = date[4:6]
                            day = date[6:8]
                            hour = date[9:11]
                            minute = date[11:13]
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
                                if cyclone_id in nc_file:
                                    nc_path = os.path.join(dayera5_path, nc_file)
                                    break
                            reg_era5 = regrid_colocs.regrid_files_era5(
                                [nc_path],
                                "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/"
                                "regrid_era5/regridded_era5",
                                resolution_km=2,
                                grid_size_km=300,
                                list_sar_path=list_sar_path,
                                index_hour=int(hour) - 1 + int(minute) // 30,
                            )[0]
                            era5_train.append(reg_era5)

                        sar_train.append(ds_aeqd["owiWindSpeed"].values)
                        irwin_train.append(sargeo["IRWIN"].values)

                    # --- Ajout des features environnementales (shear) si modèle conditionnel ---
                    if conditional_model:
                        shear_vec = row[shear_cols].to_numpy(dtype=np.float32)
                        shear_vec = np.nan_to_num(shear_vec, nan=0.0)
                        shear_vec = (shear_vec - shear_mean) / shear_std
                        infos_train.append({**{k: row[k] for k in keys}, "shear": shear_vec})
                    else:
                        infos_train.append({k: row[k] for k in keys})

                    good_rows.append(i)

                except Exception as e:
                    print(e)
                    bad_rows.append(i)

                pbar.set_postfix(good=len(good_rows), bad=len(bad_rows))
                pbar.update(1)
            pbar.close()

            # ===========================================================
            # 2.b) Ensemble VALIDATION
            # ===========================================================
            N = len(data[data["split"] == "val"])
            pbar = tqdm(total=N, desc="Checking validation files", unit="row")
            for i, row in data[data["split"] == "val"].iterrows():
                try:
                    with xr.open_dataset(row["sargeo_path"]) as sargeo:
                        if "IRWIN" not in sargeo:
                            raise KeyError("Missing IRWIN")

                    with xr.open_dataset(row["sar_aeqd_path"]) as ds_aeqd:
                        if "owiWindSpeed" not in ds_aeqd:
                            raise KeyError("Missing owiWindSpeed")

                        if self.add_era5:
                            sar_path = row["sar_aeqd_path"]
                            list_sar_path = [sar_path]
                            cyclone_id = row["cyclone_id"]
                            date = str(sar_path.split("/")[-1].split("-")[5])
                            year = date[0:4]
                            month = date[4:6]
                            day = date[6:8]
                            hour = date[9:11]
                            minute = date[11:13]
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
                                if cyclone_id in nc_file:
                                    nc_path = os.path.join(dayera5_path, nc_file)
                                    break
                            reg_era5 = regrid_colocs.regrid_files_era5(
                                [nc_path],
                                "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/"
                                "regrid_era5/regridded_era5",
                                resolution_km=2,
                                grid_size_km=300,
                                list_sar_path=list_sar_path,
                                index_hour=int(hour) - 1 + int(minute) // 30,
                            )[0]
                            era5_val.append(reg_era5)

                        sar_val.append(ds_aeqd["owiWindSpeed"].values)
                        irwin_val.append(sargeo["IRWIN"].values)

                    if conditional_model:
                        shear_vec = row[shear_cols].to_numpy(dtype=np.float32)
                        shear_vec = np.nan_to_num(shear_vec, nan=0.0)
                        shear_vec = (shear_vec - shear_mean) / shear_std
                        infos_val.append({**{k: row[k] for k in keys}, "shear": shear_vec})
                    else:
                        infos_val.append({k: row[k] for k in keys})

                    good_rows.append(i)

                except Exception:
                    bad_rows.append(i)

                pbar.set_postfix(good=len(good_rows), bad=len(bad_rows))
                pbar.update(1)
            pbar.close()

            # ===========================================================
            # 2.c) Ensemble TEST
            # ===========================================================
            N = len(data[data["split"] == "test"])
            pbar = tqdm(total=N, desc="Checking test files", unit="row")
            for i, row in data[data["split"] == "test"].iterrows():
                try:
                    with xr.open_dataset(row["sargeo_path"]) as sargeo:
                        if "IRWIN" not in sargeo:
                            raise KeyError("Missing IRWIN")

                    with xr.open_dataset(row["sar_aeqd_path"]) as ds_aeqd:
                        if "owiWindSpeed" not in ds_aeqd:
                            raise KeyError("Missing owiWindSpeed")

                        if self.add_era5:
                            sar_path = row["sar_aeqd_path"]
                            list_sar_path = [sar_path]
                            cyclone_id = row["cyclone_id"]
                            date = str(sar_path.split("/")[-1].split("-")[5])
                            year = date[0:4]
                            month = date[4:6]
                            day = date[6:8]
                            hour = date[9:11]
                            minute = date[11:13]
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
                                if cyclone_id in nc_file:
                                    nc_path = os.path.join(dayera5_path, nc_file)
                                    break
                            reg_era5 = regrid_colocs.regrid_files_era5(
                                [nc_path],
                                "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/"
                                "regrid_era5/regridded_era5",
                                resolution_km=2,
                                grid_size_km=300,
                                list_sar_path=list_sar_path,
                                index_hour=int(hour) - 1 + int(minute) // 30,
                            )[0]
                            era5_test.append(reg_era5)

                        sar_test.append(ds_aeqd["owiWindSpeed"].values)
                        irwin_test.append(sargeo["IRWIN"].values)

                    if conditional_model:
                        shear_vec = row[shear_cols].to_numpy(dtype=np.float32)
                        shear_vec = np.nan_to_num(shear_vec, nan=0.0)
                        shear_vec = (shear_vec - shear_mean) / shear_std
                        infos_test.append({**{k: row[k] for k in keys}, "shear": shear_vec})
                    else:
                        infos_test.append({k: row[k] for k in keys})

                    good_rows.append(i)

                except Exception:
                    bad_rows.append(i)

                pbar.set_postfix(good=len(good_rows), bad=len(bad_rows))
                pbar.update(1)
            pbar.close()

            # ===========================================================
            # 2.d) Jeu de test additionnel "Anggrek" (optionnel)
            # ===========================================================
            if anggrek_test:
                N = len(anggrek_csv)
                # Indices relatifs (-C/2 ... +C/2) des canaux IR temporels à charger
                indices = list(range(-(self.irwin_channels // 2), (self.irwin_channels // 2) + 1))
                pbar = tqdm(total=N, desc="Checking ANGGREK files", unit="row")
                for row_idx, row in anggrek_csv.iterrows():
                    ir_path = row["ir_path"]
                    paths = [shift_ir_path(ir_path, idx=i, step_minutes=30) for i in indices]

                    sample_imgs = []
                    ok = True

                    for path in paths:
                        try:
                            with xr.open_dataset(path) as ir_ds:
                                if "IR" not in ir_ds:
                                    ok = False
                                    break
                                arr = np.squeeze(ir_ds["IR"].values)
                                sample_imgs.append(arr)
                        except FileNotFoundError:
                            ok = False
                            break

                    # On n'ajoute le sample que si on a bien C canaux
                    if ok and len(sample_imgs) == len(indices):
                        # empile en (C, H, W)
                        irwin_anggrek.append(np.stack(sample_imgs, axis=0))

                        infos_anggrek.append(
                            {
                                "sid": row["sid"],
                                "date": row["date"],
                                "vmax": row["wind_speed (m/s)"],
                                "lat": row["lat"],
                                "lon": row["lon"],
                                "analysis_vmax_cyclobs": row["analysis_vmax_cyclobs"],
                                "vmax_cyclobs": row["vmax_cyclobs"],
                                "ibtracs_vmax": row["ibtracs_vmax"],
                                "satcon_vmax": row["satcon_vmax"],
                                "era5_vmax": row["era5_vmax"],
                            }
                        )

                        cyclone_id = anggrek_csv.iloc[row_idx]["sid"]
                        if self.add_era5:
                            date = anggrek_csv.iloc[row_idx]["date"]
                            year = date[0:4]
                            month = date[5:7]
                            day = date[8:10]
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
                                if cyclone_id in nc_file:
                                    nc_path = os.path.join(dayera5_path, nc_file)
                                    break
                            reg_era5 = regrid_colocs.regrid_files_era5(
                                [nc_path],
                                "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/"
                                "regrid_era5/regridded_era5",
                                resolution_km=2,
                                grid_size_km=300,
                                list_sar_path=list_sar_path,
                                index_hour=int(hour) - 1 + int(minute) // 30,
                            )
                            era5_anggrek.append(reg_era5)

                    pbar.set_postfix(good=len(good_rows), bad=len(bad_rows))
                    pbar.update(1)

                # --- Mise en forme finale : (N, C, H, W) ---
                if self.add_era5:
                    era5_train = np.expand_dims(era5_train, axis=1)
                    era5_val = np.expand_dims(era5_val, axis=1)
                    era5_test = np.expand_dims(era5_test, axis=1)
                    era5_anggrek = np.array(era5_anggrek)

                    print(np.array(irwin_anggrek).shape, np.array(era5_anggrek).shape)
                    print(np.array(irwin_train).shape, np.array(era5_train).shape)
                    print(np.array(irwin_val).shape, np.array(era5_val).shape)
                    print(np.array(irwin_test).shape, np.array(era5_test).shape)

                    h_era5, w_era5 = era5_anggrek.shape[-2], era5_anggrek.shape[-1]
                    h_anggrek = np.array(irwin_anggrek).shape[-2]

                    # Recadrage central des images IR Anggrek à la taille des données ERA5
                    irwin_anggrek = np.array(irwin_anggrek)[
                        :, :,
                        h_anggrek // 2 - h_era5 // 2 : h_anggrek // 2 + h_era5 // 2,
                        h_anggrek // 2 - h_era5 // 2 : h_anggrek // 2 + h_era5 // 2,
                    ]
                    irwin_anggrek = np.concatenate([irwin_anggrek, era5_anggrek], axis=1)
                else:
                    irwin_anggrek = np.array(irwin_anggrek)
                    irwin_train = np.array(irwin_train)
                    irwin_val = np.array(irwin_val)
                    irwin_test = np.array(irwin_test)
                pbar.close()

                # On ne garde dans `data` que les lignes correspondant aux fichiers
                # qui se sont bien ouverts.
                data = data.loc[good_rows].reset_index(drop=True)

                print(f"✅ Kept rows that open correctly: {len(data)}")
                print(f"❌ Dropped rows that failed: {len(bad_rows)}")

        # -------------------------------------------------------------
        # 3) Stockage des métadonnées sur l'instance
        # -------------------------------------------------------------
        if self.cfg.channel_splitting and False:
            self.infos_train = [info for info in infos_train for _ in range(9)]
            self.infos_val = [info for info in infos_val for _ in range(9)]
            self.infos_test = [info for info in infos_test for _ in range(9)]
        else:
            self.infos_train = infos_train
            self.infos_val = infos_val
            self.infos_test = infos_test
        self.infos_anggrek = infos_anggrek if anggrek_test else None

        image_channels_train, image_channels_val, image_channels_test, image_channels_anggrek = (
            [], [], [], [],
        )

        # -------------------------------------------------------------
        # 4) Fusion IRWIN + ERA5 si nécessaire
        # -------------------------------------------------------------
        if self.add_era5:
            h_sargeo = np.array(irwin_train).shape[-1]
            irwin_train = np.array(irwin_train)[
                :, :,
                h_sargeo // 2 - h_era5 // 2 : h_sargeo // 2 + h_era5 // 2,
                h_sargeo // 2 - w_era5 // 2 : h_sargeo // 2 + w_era5 // 2,
            ]
            irwin_val = np.array(irwin_val)[
                :, :,
                h_sargeo // 2 - h_era5 // 2 : h_sargeo // 2 + h_era5 // 2,
                h_sargeo // 2 - w_era5 // 2 : h_sargeo // 2 + w_era5 // 2,
            ]
            irwin_test = np.array(irwin_test)[
                :, :,
                h_sargeo // 2 - h_era5 // 2 : h_sargeo // 2 + h_era5 // 2,
                h_sargeo // 2 - w_era5 // 2 : h_sargeo // 2 + w_era5 // 2,
            ]

            irwin_train = np.concatenate([irwin_train, era5_train], axis=1)  # (N, 10, H, W) par ex.
            irwin_val = np.concatenate([irwin_val, era5_val], axis=1)
            irwin_test = np.concatenate([irwin_test, era5_test], axis=1)
            irwin_anggrek = np.array(irwin_anggrek) if anggrek_test else None

        self.sar_train, self.sar_val, self.sar_test = (
            np.array(sar_train), np.array(sar_val), np.array(sar_test),
        )

        N, _, H, W = irwin_train.shape

        # -------------------------------------------------------------
        # 5) Construction des canaux d'entrée selon le mode `input_data`
        # -------------------------------------------------------------
        if self.input_data == "all_channels":
            # Mode "tous les canaux" : on garde les 9 canaux IRWIN temporels bruts.
            for i in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                image_channels_train.append(irwin_train[:, i, :, :])
                image_channels_val.append(irwin_val[:, i, :, :])
                image_channels_test.append(irwin_test[:, i, :, :])
                if anggrek_test:
                    image_channels_anggrek.append(irwin_anggrek)

        elif self.input_data == "normal":

            def transform_irwin_channels(x):
                """
                Réduit les 10 canaux (9 IRWIN + 1 ERA5) à 6 canaux par
                moyennage temporel par fenêtres glissantes.

                x shape: (N, 10, W, H)
                return shape: (N, 6, W, H)
                """
                c0 = x[:, 0:1, :, :]                              # garder canal 0
                c1 = x[:, 0:5, :, :].mean(axis=1, keepdims=True)  # mean(0,1,2,3,4)
                c2 = x[:, 2:5, :, :].mean(axis=1, keepdims=True)  # mean(2,3,4)
                c3 = x[:, 4:7, :, :].mean(axis=1, keepdims=True)  # mean(4,5,6)
                c4 = x[:, 4:9, :, :].mean(axis=1, keepdims=True)  # mean(4,5,6,7,8)
                c5 = x[:, 9:10, :, :]                             # garder canal 9
                return np.concatenate([c0, c1, c2, c3, c4, c5], axis=1)

            # Sélection des canaux IRWIN centrés autour du canal 4, +/- irwin_channels/2
            if not self.add_era5:
                irwin_train = irwin_train[:, 4 - self.irwin_channels // 2 : 4 + self.irwin_channels // 2 + 1, :, :]
                irwin_val = irwin_val[:, 4 - self.irwin_channels // 2 : 4 + self.irwin_channels // 2 + 1, :, :]
                irwin_test = irwin_test[:, 4 - self.irwin_channels // 2 : 4 + self.irwin_channels // 2 + 1, :, :]
            else:
                irwin_train = np.concatenate(
                    [irwin_train[:, 4 - self.irwin_channels // 2 : 4 + self.irwin_channels // 2 + 1, :, :], era5_train],
                    axis=1,
                )
                irwin_val = np.concatenate(
                    [irwin_val[:, 4 - self.irwin_channels // 2 : 4 + self.irwin_channels // 2 + 1, :, :], era5_val],
                    axis=1,
                )
                irwin_test = np.concatenate(
                    [irwin_test[:, 4 - self.irwin_channels // 2 : 4 + self.irwin_channels // 2 + 1, :, :], era5_test],
                    axis=1,
                )

            image_channels_train.append(transform_irwin_channels(irwin_train) if self.ir_smoothing else irwin_train)
            image_channels_val.append(transform_irwin_channels(irwin_val) if self.ir_smoothing else irwin_val)
            image_channels_test.append(transform_irwin_channels(irwin_test) if self.ir_smoothing else irwin_test)

            if anggrek_test:
                image_channels_anggrek.append(
                    transform_irwin_channels(irwin_anggrek) if self.ir_smoothing else irwin_anggrek
                )

        elif self.input_data == "normal+gradients":
            # Canal central + ses gradients spatiaux (lignes/colonnes)
            image_channels_train.append(irwin_train[:, 4, :, :])
            image_channels_val.append(irwin_val[:, 4, :, :])
            image_channels_test.append(irwin_test[:, 4, :, :])
            if anggrek_test:
                image_channels_anggrek.append(irwin_anggrek)

            image_channels_train.append(np.gradient(irwin_train[:, 4, :, :])[0])
            image_channels_train.append(np.gradient(irwin_train[:, 4, :, :])[1])

            image_channels_val.append(np.gradient(irwin_val[:, 4, :, :])[0])
            image_channels_val.append(np.gradient(irwin_val[:, 4, :, :])[1])

            image_channels_test.append(np.gradient(irwin_test[:, 4, :, :])[0])
            image_channels_test.append(np.gradient(irwin_test[:, 4, :, :])[1])
            if anggrek_test:
                image_channels_anggrek.append(np.gradient(irwin_anggrek)[0])
                image_channels_anggrek.append(np.gradient(irwin_anggrek)[1])

        elif self.input_data == "normal mean":
            # Moyenne temporelle de tous les canaux IRWIN
            image_channels_train.append(np.nanmean(irwin_train, axis=1))
            image_channels_val.append(np.nanmean(irwin_val, axis=1))
            image_channels_test.append(np.nanmean(irwin_test, axis=1))
            if anggrek_test:
                image_channels_anggrek.append(irwin_anggrek)

        # Empile la liste de canaux en un seul tenseur (N, C, H, W)
        self.X_train = np.concatenate(image_channels_train, axis=1)
        self.X_val = np.concatenate(image_channels_val, axis=1)
        self.X_test = np.concatenate(image_channels_test, axis=1)
        if anggrek_test:
            self.X_anggrek = np.array(image_channels_anggrek).squeeze(axis=0)

        print("Start Reshaping Data ........")

        # -------------------------------------------------------------
        # 6) Downsampling optionnel (2km -> 4km)
        # -------------------------------------------------------------
        if self.cfg.downsampling:
            print("Start downsampling data to 4Km:")

            self.X_train = dataprep.downsample_2km_to_4km(self.X_train)
            self.X_val = dataprep.downsample_2km_to_4km(self.X_val)
            self.X_test = dataprep.downsample_2km_to_4km(self.X_test)

            if anggrek_test:
                self.X_anggrek = dataprep.downsample_2km_to_4km(self.X_anggrek)

            self.sar_train = dataprep.downsample_2km_to_4km(self.sar_train)
            self.sar_val = dataprep.downsample_2km_to_4km(self.sar_val)
            self.sar_test = dataprep.downsample_2km_to_4km(self.sar_test)

            print("Shape of the Train Input after downsampling:", self.X_train.shape, "and output:", self.sar_train.shape)
            print("Shape of the Val Input after downsampling:", self.X_val.shape, "and output:", self.sar_val.shape)
            print("Shape of the Test Input after downsampling:", self.X_test.shape, "and output:", self.sar_test.shape)

        # -------------------------------------------------------------
        # 7) Recadrage spatial centré à la taille `size`
        # -------------------------------------------------------------
        N, C, H, W = self.X_train.shape
        print(N, C, H, W)

        if anggrek_test:
            N, C, h_anggrek, W_anggrek = self.X_anggrek.shape
        N, H_sar, W_sar = self.sar_train.shape
        print(N, H_sar, W_sar)

        self.X_train = (
            self.X_train[:, :, H // 2 - size // 2 : H // 2 + size // 2, W // 2 - size // 2 : W // 2 + size // 2]
            if not self.ir_smoothing
            else dataprep.build_irwin_channels(
                self.X_train[:, :, H // 2 - size // 2 : H // 2 + size // 2, W // 2 - size // 2 : W // 2 + size // 2], 9
            )
        )
        self.X_val = (
            self.X_val[:, :, H // 2 - size // 2 : H // 2 + size // 2, W // 2 - size // 2 : W // 2 + size // 2]
            if not self.ir_smoothing
            else dataprep.build_irwin_channels(
                self.X_val[:, :, H // 2 - size // 2 : H // 2 + size // 2, W // 2 - size // 2 : W // 2 + size // 2], 9
            )
        )
        self.X_test = (
            self.X_test[:, :, H // 2 - size // 2 : H // 2 + size // 2, W // 2 - size // 2 : W // 2 + size // 2]
            if not self.ir_smoothing
            else dataprep.build_irwin_channels(
                self.X_test[:, :, H // 2 - size // 2 : H // 2 + size // 2, W // 2 - size // 2 : W // 2 + size // 2], 9
            )
        )
        if anggrek_test:
            self.X_anggrek = (
                self.X_anggrek[
                    :, :,
                    h_anggrek // 2 - size // 2 : h_anggrek // 2 + size // 2,
                    W_anggrek // 2 - size // 2 : W_anggrek // 2 + size // 2,
                ]
                if not self.ir_smoothing
                else dataprep.build_irwin_channels(
                    self.X_anggrek[
                        :, :,
                        h_anggrek // 2 - size // 2 : h_anggrek // 2 + size // 2,
                        W_anggrek // 2 - size // 2 : W_anggrek // 2 + size // 2,
                    ],
                    9,
                )
            )

        self.sar_train = self.sar_train[:, H_sar // 2 - size // 2 : H_sar // 2 + size // 2, W_sar // 2 - size // 2 : W_sar // 2 + size // 2]
        self.sar_val = self.sar_val[:, H_sar // 2 - size // 2 : H_sar // 2 + size // 2, W_sar // 2 - size // 2 : W_sar // 2 + size // 2]
        self.sar_test = self.sar_test[:, H_sar // 2 - size // 2 : H_sar // 2 + size // 2, W_sar // 2 - size // 2 : W_sar // 2 + size // 2]

        # -------------------------------------------------------------
        # 8) Conversion Kelvin -> Celsius pour tous les canaux IR
        # -------------------------------------------------------------
        n_ir_channels = self.X_train.shape[1]
        for c in range(n_ir_channels):
            self.X_train[:, c] = self.X_train[:, c] - 273.15
            self.X_val[:, c] = self.X_val[:, c] - 273.15
            self.X_test[:, c] = self.X_test[:, c] - 273.15
            if anggrek_test:
                self.X_anggrek[:, c] = self.X_anggrek[:, c] - 273.15

        # -------------------------------------------------------------
        # 9) Découpage par canal ("channel splitting") si activé
        # -------------------------------------------------------------
        if self.cfg.channel_splitting:
            self.X_train, self.sar_train = dataprep.split_ir_channels_and_repeat_sar(self.X_train, self.sar_train)
            if anggrek_test:
                self.X_anggrek = self.X_anggrek[:, n_ir_channels // 2, :, :]

            self.X_val, self.sar_val = self.X_val[:, n_ir_channels // 2, :, :], self.sar_val
            self.X_test, self.sar_test = self.X_test[:, n_ir_channels // 2, :, :], self.sar_test

            N, H, W = self.X_anggrek.shape
            self.X_anggrek = self.X_anggrek.reshape((N, 1, W, H))
            N_val = self.X_val.shape[0]
            N_test = self.X_test.shape[0]
            self.X_val = self.X_val.reshape((N_val, 1, W, H))
            self.X_test = self.X_test.reshape((N_test, 1, W, H))

        print("Final Shape of train Input is ", self.X_train.shape)
        print("Final Shape of train Output is ", self.sar_train.shape)
        print("Final Shape of Validation Input is ", self.X_val.shape)
        print("Final Shape of Validation Output is ", self.sar_val.shape)
        print("Final Shape of Test Input is ", self.X_test.shape)
        print("Final Shape of Test Output is ", self.sar_test.shape)
        print("Final Shape of Anggrek Input is ", self.X_anggrek.shape)

        # -------------------------------------------------------------
        # 10) Masque des pixels SAR valides (non-NaN)
        # -------------------------------------------------------------
        self.mask_train = np.isfinite(self.sar_train).astype(np.float32)
        self.mask_val = np.isfinite(self.sar_val).astype(np.float32)
        self.mask_test = np.isfinite(self.sar_test).astype(np.float32)

        # -------------------------------------------------------------
        # 11) Remplacement des NaN / Inf par 0
        # -------------------------------------------------------------
        self.X_train = np.nan_to_num(self.X_train, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_val = np.nan_to_num(self.X_val, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_test = np.nan_to_num(self.X_test, nan=0.0, posinf=0.0, neginf=0.0)
        if anggrek_test:
            self.X_anggrek = np.nan_to_num(self.X_anggrek, nan=0.0, posinf=0.0, neginf=0.0)

        self.sar_train = np.nan_to_num(self.sar_train, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar_val = np.nan_to_num(self.sar_val, nan=0.0, posinf=0.0, neginf=0.0)
        self.sar_test = np.nan_to_num(self.sar_test, nan=0.0, posinf=0.0, neginf=0.0)

        # -------------------------------------------------------------
        # 12) Calcul du moment angulaire (AAM) si demandé en sortie
        # -------------------------------------------------------------
        if self.output_data == "aam":
            self.sar_train = dataprep.create_moment_sar(self.sar_train)
            self.sar_val = dataprep.create_moment_sar(self.sar_val)
            self.sar_test = dataprep.create_moment_sar(self.sar_test)

        # Sauvegarde de contrôle des données IR test / Anggrek
        with open("test_ir.pkl", "wb") as f:
            pkl.dump(self.X_test, f)
        with open("anggrek_ir.pkl", "wb") as f:
            pkl.dump(self.X_anggrek, f)

        # -------------------------------------------------------------
        # 13) Augmentation de données (uniquement sur le train)
        # -------------------------------------------------------------
        if self.augmentation:
            print("Start Data Augmentation for train set : ----------")
            self.X_train, self.sar_train, self.mask_train, self.infos_train = dataprep.data_augmentation(
                self.X_train, self.sar_train, self.mask_train, self.infos_train
            )
        print("New train size after augmentation is ", len(self.X_train))

        # -------------------------------------------------------------
        # 14) Graphiques de contrôle qualité
        # -------------------------------------------------------------
        print("Plot Data distribution : --------------------")
        dataprep.plot_metric_scatter(
            true_values=[d["vmax"] for d in self.infos_train],
            pred_values=[d["analysis_vmax"] for d in self.infos_train],
            output_path=self.target_dir,
            file_name="analysis_vmax_and_vmax_comparaison_train",
            title="analysis vmax and vmax comparaison in the train set",
            xlabel="vmax (m\\s)",
            ylabel="analysis_vmax (m\\s)",
        )
        dataprep.plot_metric_scatter(
            true_values=[d["vmax"] for d in self.infos_val],
            pred_values=[d["analysis_vmax"] for d in self.infos_val],
            output_path=self.target_dir,
            file_name="analysis_vmax_and_vmax_comparaison_val",
            title="analysis vmax and vmax comparaison in the val set",
            xlabel="vmax (m\\s)",
            ylabel="analysis_vmax (m\\s)",
        )
        dataprep.plot_metric_scatter(
            true_values=[d["vmax"] for d in self.infos_test],
            pred_values=[d["analysis_vmax"] for d in self.infos_test],
            output_path=self.target_dir,
            file_name="analysis_vmax_and_vmax_comparaison_test",
            title="analysis vmax and vmax comparaison in the test set",
            xlabel="vmax (m\\s)",
            ylabel="analysis_vmax (m\\s)",
        )
        dataprep.plot_rmax_distribution(
            infos_train=self.infos_train,
            infos_val=self.infos_val,
            output_path=self.target_dir,
            file_name="analysis_rmax_distribution_train_vs_val.png",
        )

        # -------------------------------------------------------------
        # 15) Normalisation des entrées (z-score canal par canal)
        # -------------------------------------------------------------
        print("Start Normalisation......")
        mean_x, std_x = [], []
        for c in range(self.X_train.shape[1]):
            self.X_train[:, c], mean, std = dataprep.z_score(self.X_train[:, c])
            mean_x.append(mean)
            std_x.append(std)
        self.mean_X, self.std_X = mean_x, std_x

        for c in range(self.X_train.shape[1]):
            self.X_val[:, c], _, _ = dataprep.z_score(self.X_val[:, c], mean_value=mean_x[c], std_value=std_x[c])
            self.X_test[:, c], _, _ = dataprep.z_score(self.X_test[:, c], mean_value=mean_x[c], std_value=std_x[c])
            if anggrek_test:
                self.X_anggrek[:, c], _, _ = dataprep.z_score(
                    self.X_anggrek[:, c], mean_value=mean_x[c], std_value=std_x[c]
                )

        # -------------------------------------------------------------
        # 16) Transformation logarithmique optionnelle du vent SAR
        # -------------------------------------------------------------
        if self.log_wind:
            self.sar_train = np.log(self.sar_train + 1e-10)
            self.sar_val = np.log(self.sar_val + 1e-10)
            self.sar_test = np.log(self.sar_test + 1e-10)

        # -------------------------------------------------------------
        # 17) Normalisation de la sortie SAR (z-score ou annulaire)
        # -------------------------------------------------------------
        if norm == "z_score":
            self.sar_train, self.mean_sar, self.std_sar = dataprep.z_score(self.sar_train)
            self.sar_val, _, _ = dataprep.z_score(self.sar_val, mean_value=self.mean_sar, std_value=self.std_sar)
            self.sar_test, _, _ = dataprep.z_score(self.sar_test, mean_value=self.mean_sar, std_value=self.std_sar)

        elif norm == "annular":
            self.sar_train, stats = dataprep.annular_normalization(self.sar_train, bin_size=1, mask=None)
            self.mean_sar = stats["mean"]
            self.std_sar = stats["std"]
            self.sar_val, _ = dataprep.annular_normalization(self.sar_val, bin_size=1, mask=None, stats=stats)
            self.sar_test, _ = dataprep.annular_normalization(self.sar_test, mask=None, stats=stats)

        # -------------------------------------------------------------
        # 18) Sauvegarde des statistiques de normalisation
        # -------------------------------------------------------------
        out_stats_dir = os.path.join(self.target_dir, "stats_normalisation", "OUTPUT")
        os.makedirs(out_stats_dir, exist_ok=True)

        with open(os.path.join(out_stats_dir, "stats.pkl"), "wb") as f:
            pkl.dump(
                {
                    "normalisation_type": str(norm),
                    "std_sar": self.std_sar,
                    "mean_sar": self.mean_sar,
                    "mean_x": mean_x,
                    "std_x": std_x,
                },
                f,
            )

        # -------------------------------------------------------------
        # 19) Sauvegarde des données Anggrek normalisées
        # -------------------------------------------------------------
        if anggrek_test:
            anggrek_dir = os.path.join(self.target_dir, "stats_normalisation", "Anggrek_data_normalised")
            os.makedirs(anggrek_dir, exist_ok=True)

            with open(os.path.join(anggrek_dir, "data.pkl"), "wb") as f:
                pkl.dump(
                    {
                        "IR_anggrek": self.X_anggrek,
                        "infos_anggrek": self.infos_anggrek,
                    },
                    f,
                )

        print("SAR normalized.")