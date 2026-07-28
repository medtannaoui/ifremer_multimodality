import numpy as np
import xarray as xr
import os
import pickle as pkl

def generate_sequence_irar_temporal_mode(all_sequences, enu):

    cyclone_id = all_sequences[enu]["cyc_id"]
    sequence = all_sequences[enu]["sequence"]
    ir_sequence, wind_sequence = [], []
    wind_mask = []
    for tmp in sequence:
        path = tmp["path"]
        try : 
            with xr.open_dataset(path, engine="netcdf4") as ds:
                ir_sequence.append(ds["ir_aeqd"].values)
                if tmp["has_wind"] : 
                    wind_sequence.append(ds["wind_aeqd"].values)
                    wind_mask.append(True)
                else : 
                    wind_sequence.append(
                                        np.zeros((501, 501), dtype=np.float32)
                                        )
                    wind_mask.append(False)
        except Exception as e: 
            break
    
    if len(ir_sequence) != 12:
        return None
    if not any(wind_mask):
        return None

    ir_sequence = np.stack(ir_sequence,axis=0) - 273.15 # (12, 501, 501)   (K to °C)
    wind_sequence = np.stack(wind_sequence,axis=0)
    wind_mask = np.asarray(wind_mask,dtype=bool)
    return ir_sequence, wind_sequence, wind_mask

def centrage_sur_imagesize_irar_temporale_mode(cfg, target_dir, irwin_train, sar_train,  irwin_val, sar_val,  irwin_test, sar_test ):
    W, H = irwin_train.shape[-2:]
    start_h = H // 2 - cfg.img_size // 2
    end_h = H // 2 + cfg.img_size // 2
    start_w = W // 2 - cfg.img_size // 2
    end_w = W // 2  + cfg.img_size // 2
    # Train
    X_train = irwin_train[:, :, start_w:end_w, start_h:end_h]
    sar_train = sar_train[:, :, start_w:end_w, start_h:end_h]
    # Validation
    X_val = irwin_val[:, :, start_w:end_w, start_h:end_h]
    sar_val = sar_val[:, :, start_w:end_w, start_h:end_h]
    # Test
    X_test = irwin_test[:, :, start_w:end_w, start_h:end_h]
    sar_test = sar_test[:, :, start_w:end_w, start_h:end_h]

    print("Train")
    print(X_train.shape)
    print(sar_train.shape)
    print("\nValidation")
    print(X_val.shape)
    print(sar_val.shape)
    print("\nTest")
    print(X_test.shape)
    print(sar_test.shape)
    try:
        with open(os.path.join(target_dir,"sequence_data.pkl"),"wb") as f:
            pkl.dump({"x_test":X_test,
                    "y_test":sar_test}
                    , f)
    except Exception as e : 
        print("erreur dans la sauvgarde des sequences :",e)
    return X_train,sar_train, X_val, sar_val, X_test, sar_test
    
def create_temporla_mask(sar_train, sar_val, sar_test, wind_mask_train, wind_mask_val, wind_mask_test):
    # Spatial mask (NaN)
    mask_train = np.isfinite(sar_train).astype(np.float32)
    mask_val   = np.isfinite(sar_val).astype(np.float32)
    mask_test  = np.isfinite(sar_test).astype(np.float32)
    # Temporal mask -> (N, 12, 1, 1)
    train_temporal = wind_mask_train[:, :, None, None].astype(np.float32)
    val_temporal   = wind_mask_val[:, :, None, None].astype(np.float32)
    test_temporal  = wind_mask_test[:, :, None, None].astype(np.float32)
    # Final mask -> (N, 12, H, W)
    mask_train *= train_temporal
    mask_val   *= val_temporal
    mask_test  *= test_temporal
    return mask_train, mask_val, mask_test

def print_infos_temporal_data(C, mask_train, N_train,):
    # Affichage de la disponibilité SAR par canal
    for c in range(C):
        n_valid_pixels = mask_train[:, c].sum()

        n_valid_sequences = np.sum(
            mask_train[:, c]
            .reshape(N_train, -1)
            .sum(axis=1) > 0
        )
        print(
            f"Canal SAR {c}: "
            f"pixels valides={n_valid_pixels}, "
            f"séquences avec SAR={n_valid_sequences}"
        )

def normalize_sar_temporal_mode(dataprep, sar_train, sar_val, sar_test, mask_train, mask_val, mask_test):
    N_train, C, H, W = sar_train.shape
    N_val = sar_val.shape[0]
    N_test = sar_test.shape[0]
    sar_train_flat = sar_train.reshape(
        N_train * C,
        H,
        W,
    )
    mask_train_flat = mask_train.reshape(
        N_train * C,
        H,
        W,
    ).astype(bool)

    sar_val_flat = sar_val.reshape(
        N_val * C,
        H,
        W,
    )
    mask_val_flat = mask_val.reshape(
        N_val * C,
        H,
        W,
    ).astype(bool)

    sar_test_flat = sar_test.reshape(
        N_test * C,
        H,
        W,
    )
    mask_test_flat = mask_test.reshape(
        N_test * C,
        H,
        W,
    ).astype(bool)

    # Vérification : au moins une observation SAR valide
    if not mask_train_flat.any():
        raise RuntimeError(
            "Aucun pixel SAR valide dans le jeu d'entraînement."
        )

    # Calcul d'UNE statistique par anneau avec tous les canaux
    sar_train_flat, mean_sar, std_sar = dataprep.z_score(
        sar_train_flat,
        mask = mask_train_flat
    )
    # calculées uniquement sur le train
    sar_val_flat, _, _ = dataprep.z_score(
        sar_val_flat,
        mean_value= mean_sar,
        std_value = std_sar,
        mask=mask_val_flat
    )
    sar_test_flat, _, _ = dataprep.z_score(
        sar_test_flat,
        mean_value = mean_sar,
        std_value = std_sar,
        mask = mask_test_flat
    )
    # Retour aux formes temporelles originales
    sar_train = sar_train_flat.reshape(
        N_train,
        C,
        H,
        W,
    ).astype(np.float32)
    sar_val = sar_val_flat.reshape(
        N_val,
        C,
        H,
        W,
    ).astype(np.float32)
    sar_test = sar_test_flat.reshape(
        N_test,
        C,
        H,
        W,
    ).astype(np.float32)
    # Remise à zéro des pixels et canaux sans SAR
    sar_train *= mask_train
    sar_val *= mask_val
    sar_test *= mask_test
    mean_sar = mean_sar
    std_sar = std_sar
    
    return {
        "mean_sar": mean_sar,
        "std_sar": std_sar,
    }
