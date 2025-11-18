# This script wil be used to test a simple U-NET for a regression simple (IR -> SAR)

import os
import sys
os.chdir("/scale/user/mtannaou/alternance")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

print("Python Path:", sys.path)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys


print("---",os.getcwd())
from importlib import reload
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle as pkl


import src.IR_to_SAR.data_preprocessing as dataprep
reload(dataprep)
print("dataprep updated")


from src.IR_to_SAR.ML_IR_SAR.model import create_model
from src.IR_to_SAR.ML_IR_SAR.callbacks import EarlyStopping,ModelCheckpoint


# ============================================================
# ===============  DATASET  =================================
# ============================================================

class IRSARDataset(Dataset):
    """
    Dataset for IR → SAR prediction.
    IR shape:  (N, H_ir, W_ir)
    SAR shape: (N, H_ir, W_ir)
    """
    def __init__(self):
        with open("/scale/user/mtannaou/alternance/src/IR_to_SAR/data_sar_ir_pkl/irwin_wind_tensors.pkl","rb") as f :
            data_ir_sar = pkl.load(f)
        
        self.ir = data_ir_sar[list(data_ir_sar.keys())[0]]
        
        self.sar = data_ir_sar[list(data_ir_sar.keys())[1]]
        self.ir, self.min_val_ir, self.max_val_ir = dataprep.min_max_normalize_numpy(self.ir)
        self.sar , self.min_val_sar, self.max_val_sar = dataprep.min_max_normalize_numpy(self.sar)
        print(self.ir.shape,self.sar.shape)
        print("Data collected and normalized")
        

    def __len__(self):    #number of observations
        return len(self.ir)

    def __getitem__(self, idx):             # retrun an example of index idx a,d add a channel axis (H,W) to (1,H,W)
        ir  = torch.tensor(self.ir[idx], dtype=torch.float32).unsqueeze(0) 
        sar = torch.tensor(self.sar[idx], dtype=torch.float32).unsqueeze(0)
        return ir, sar



# ============================================================
# ===============  TRAINING FUNCTIONS  =======================
# ============================================================

def train_one_epoch(model, dataloader, optimizer, loss_fn, metrics=None, device='cpu'):
    """
    Train the model for one epoch using IR -> SAR regression.
    
    Args:
        model: PyTorch model (UNet, Diffusion U-Net, etc.)
        dataloader: DataLoader providing (ir, sar)
        optimizer: optimizer instance
        loss_fn: loss function (MSE, MAE, etc.)
        metrics: optional metrics class with reset(), update(), compute()
        device: 'cpu' or 'cuda'
    """
    model.train()
    total_loss = 0
    if metrics:
        metrics.reset()

    for ir, sar in tqdm(dataloader, desc="Training"):
        ir = ir.to(device)
        sar = sar.to(device)

        optimizer.zero_grad()

        # ---- Model Forward (Diffusion UNet requires timesteps) ----
        pred = model(ir, timestep=0).sample  

        # ---- Loss & Backprop ----
        loss = loss_fn(pred, sar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ---- Update Metrics ----
        if metrics:
            metrics.update(pred.detach(), sar.detach())

    avg_loss = total_loss / len(dataloader)
    computed_metrics = metrics.compute() if metrics else {}

    return avg_loss, computed_metrics


def validate(model, dataloader, loss_fn, metrics=None, device='cpu'):
    """
    Validate the model (IR -> SAR regression) without gradient updates.
    """
    model.eval()
    total_loss = 0
    if metrics:
        metrics.reset()

    with torch.no_grad():
        for ir, sar in tqdm(dataloader, desc="Validating"):
            ir = ir.to(device)
            sar = sar.to(device)

            # Forward
            pred = model(ir, timestep=0).sample

            # Loss
            loss = loss_fn(pred, sar)
            total_loss += loss.item()

            # Metrics
            if metrics:
                metrics.update(pred, sar)

    avg_loss = total_loss / len(dataloader)
    computed_metrics = metrics.compute() if metrics else {}

    return avg_loss

# ============================================================
# =======================   MAIN   ===========================
# ============================================================

def main(
    num_epochs=50,
    batch_size=8,
    lr=1e-3,
    patience=10,
    min_delta=0.0,
    save_dir="/scale/user/mtannaou/alternance/src/IR_to_SAR/train_pt_result",
    test=False
):

    # ---------------------------
    # TEST DATA (if needed)
    # ---------------------------
    if test:
        N, H, W = 200, 64, 64
        ir_array  = np.random.rand(N, H, W)
        sar_array = np.random.rand(N, H, W)

    # ---------------------------
    # Dataset & DataLoader
    # ---------------------------
    dataset = IRSARDataset()   # two tensors (N,Hir,Wir) et (N,H_sar,W_sar)
    #split into train val test
    dictio = dataprep.train_val_test_split(dataset.ir,dataset.sar,train_size= 0.6,val_size=0.2,test_size=0.2)


    dataloader_train = DataLoader(dictio["train"], batch_size=batch_size, shuffle=True) # batchs
    dataloader_val = DataLoader(dictio["val"], batch_size=batch_size, shuffle=True)


    # ---------------------------
    # Model setup
    # ---------------------------
    print("Model Creation")
    model = create_model().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ---------------------------
    # Early stopping & checkpoint
    # ---------------------------
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_unet.pth")
    best_val_loss = float("inf")

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(num_epochs):
        print(f"\n==============================")
        print(f"  Epoch {epoch+1}/{num_epochs}")
        print(f"==============================")

        train_loss = train_one_epoch(model, dataloader_train, optimizer, loss_fn)
        val_loss, val_mae = validate(model, dataloader_val, loss_fn)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print(f"Val   MAE : {val_mae:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # ---- Save best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Checkpoint saved → {best_model_path}")

        # ---- Early Stopping ----
        if stopper.step(val_loss):
            break

    print("\n✔ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


# Run training
if __name__ == "__main__":
    main(num_epochs=5,
         batch_size=16,
         lr=1e-3,
         test=False)


# main(
#     num_epochs=50,
#     batch_size=8,
#     lr=1e-3,
#     patience=10,
#     min_delta=0.0,
#     save_dir="/scale/user/mtannaou/alternance/src/ML_SAR/train_pt_results",
#     test=False
# )