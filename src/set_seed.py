# Thsi script will be used for fixing the seed of all libraries
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Pour des résultats reproductibles sur GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Active le seed global
set_seed(42)
