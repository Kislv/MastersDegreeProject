import torch

TORCH_TENSORS_KEYWOED:str = 'pt'
CUDA_KEYWORD:str = 'cuda'
CPU_KEYWORD:str = 'cpu'
MOST_EFFECTIVE_AVAILABLE_DEVICE:bool = torch.device(CUDA_KEYWORD if torch.cuda.is_available() else CPU_KEYWORD)

ATTENTION_MASK_KEYWORD:str = 'attention_mask'
SKLEARN_MULTINOMIAL_KEYWORD:str = 'multinomial'
