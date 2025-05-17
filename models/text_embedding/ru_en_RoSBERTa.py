import torch
import sys

sys.path.append('../..')
from models.config import (
    MOST_EFFECTIVE_AVAILABLE_DEVICE,
)

DEVICE:torch.device = MOST_EFFECTIVE_AVAILABLE_DEVICE

NORMALIZE_P:int = 2
NORMALIZE_DIM:int = 1
CLAMP_MIN:float = 1e-9
