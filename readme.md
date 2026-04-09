# 🌪️ IR → SAR Translation using Regression and Flow Matching

## 📌 Overview

This project focuses on learning a mapping from **Infrared (IR) satellite data** to **Synthetic Aperture Radar (SAR)** observations for tropical cyclones.

The goal is to:

* Reconstruct SAR-like wind structures from IR imagery
* Capture fine-scale cyclone dynamics (e.g. eyewall, asymmetries)
* Improve predictions using **Flow Matching** and **Residual Flow Matching**

---

## 🚀 Features

* ✅ **Supervised regression (U-Net)**: IR → SAR baseline
* 🔁 **Flow Matching (FM)**: stochastic generation of SAR fields
* ➕ **Residual Flow Matching**: learns residuals on top of regression
* 📊 Advanced losses:

  * Pixel-wise loss (weighted)
  * Gradient loss
  * Radial / structural loss
* 📈 Visualization callbacks:

  * Prediction maps
  * Ensemble mean & uncertainty (std maps)
* ⚡ Lightning Fabric training loop (multi-GPU ready)

---

## 🧠 Models

### 1. Regression Model

* U-Net architecture
* Deterministic mapping:
  `IR → SAR`

---

### 2. Flow Matching (FM)

* Learns velocity field in latent space
* Generates SAR samples from noise:

  ```
  z → SAR
  ```

---

### 3. Residual Flow Matching

* Combines regression + generative modeling:

  ```
  SAR = Regression(IR) + Residual
  ```
* Residual is modeled with flow matching

---

## 📂 Project Structure

```
.
├── src/
│   └── IR_to_SAR/
│       ├── ML_IR_SAR/
│       │   ├── train.py
│       │   ├── model.py
│       │   ├── losses.py
│       │   ├── callbacks.py
│       │   └── config.py
│       ├── data_preparation/
│       └── flow_matching/
│
├── data_notebook/
├── best_result_so_far/
├── wind_v02_with_era5/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/medtannaoui/ifremer_multimodality
cd ifremer_multimodality

pip install -r requirements.txt
```

---

## 🧪 Training

### 1. Configure experiment

Edit:

```
src/IR_to_SAR/ML_IR_SAR/config.yaml
```

Key options:

```yaml
use_flow_matching: false
use_residu: false

# Training
batch_size: 16
num_epochs: 1000
learning_rate: 1e-4

# Loss weights
w_pix: 1.0
w_grad: 0.1
w_radial: 0.1
```

---

### 2. Run training

```bash
python src/IR_to_SAR/ML_IR_SAR/train.py
```

---

## 🔄 Modes

### Regression

```yaml
use_flow_matching: false
```

---

### Flow Matching

```yaml
use_flow_matching: true
use_residu: false
```

---

### Residual Flow Matching

```yaml
use_flow_matching: true
use_residu: true
```

⚠️ Requires a pretrained regression model.

---

## 📊 Outputs

Each training run creates:

```
train_ir_sar_X/
├── best_regression_model.pt
├── best_fm_model.pt
├── best_fm_resid_model.pt
├── loss_history.png
├── training_history.csv
├── validation_plots/
└── config.yaml
```

---

## 📈 Evaluation & Visualization

The callbacks automatically generate:

* 🔵 Ground truth SAR
* 🟠 Regression prediction
* 🟢 Flow Matching prediction
* 🔴 Residual FM prediction
* 🌈 Uncertainty maps (ensemble std)

---

## 🧪 Ensemble Generation

Flow Matching supports stochastic sampling:

```python
ensemble = generate_ensemble(model, ir_input, n_members=20)
mean = ensemble.mean(0)
std  = ensemble.std(0)
```

---

## ⚠️ Known Pitfalls

* Ensure consistent **input channels** between training and loading models
* Residual FM requires:

  * Pretrained regression model
  * Residual statistics (mean/std)
* Always handle tensors correctly:

  ```python
  x.detach().cpu().numpy()
  ```

---

