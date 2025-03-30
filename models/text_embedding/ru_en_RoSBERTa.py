import torch
import sys

sys.path.append('../..')
from models.config import (
    CUDA_KEYWORD,
    CPU_KEYWORD,
)

DEVICE:torch.device = torch.device(CUDA_KEYWORD if torch.cuda.is_available() else CPU_KEYWORD)

NORMALIZE_P:int = 2
NORMALIZE_DIM:int = 1
CLAMP_MIN:float = 1e-9
