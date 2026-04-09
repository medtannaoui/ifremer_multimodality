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
