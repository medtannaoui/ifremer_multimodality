# src/IR_to_SAR/ML_IR_SAR/config.py

import yaml
from dataclasses import dataclass

@dataclass
class IR_SAR_Config:
    img_size: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    train_split: float
    val_split: float
    test_split: float
    data_path: str
    save_dir: str
    early_stop_patience: int
    early_stop_delta: float
    start_epoch: int
    plot_interval: int
    accelerator: str
    devices: str | int
    scheduler_every_n_epochs: int
    norm : str
    num_val_exemples : int
    barycenter : str
    augmentation : bool
    drop_nan_sar : bool
    dropout : float
    in_channels : int
    out_channels : int
    input_channels: list[str]
    @staticmethod
    def from_yaml(path: str):
        with open(path, "r") as file:
            config_data = yaml.safe_load(file)
        return IR_SAR_Config(**config_data)
