# src/IR_to_SAR/ML_IR_SAR/config.py

import yaml
from dataclasses import dataclass

@dataclass
class IR_SAR_Config:
    img_size: int
    batch_size: int
    learning_rate: float
    num_epochs: int
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
    augmentation : bool
    dropout : float
    in_channels : int
    out_channels : int
    w_pix : float
    w_grad:float
    w_radial: float
    output_data : str
    block_out_channels : list[int]
    down_block_types : list[str]
    up_block_types : list[str]
    scheduler : str
    conditional_model : bool
    anggrek_test:bool
    cross_attention_dim:int
    combined_loss : bool
    cond_dim : int
    irwin_channels : int
    add_era5 : bool
    use_flow_matching : bool
    fm_num_inference_steps : int
    fm_lr : float
    code_test : bool
    use_residu : bool
    best_regression_model_pt : str
    channel_dropout: bool
    channel_drop_prob: float
    min_keep_channels: int
    protect_channels: list[int]
    irar : bool
    temporal_mode : bool
    overlap : bool
    stride : int
    add_mw : bool
    @staticmethod
    def from_yaml(path: str):
        with open(path, "r") as file:
            config_data = yaml.safe_load(file)
        return IR_SAR_Config(**config_data)