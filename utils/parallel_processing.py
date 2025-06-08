import numpy as np
from typing import (
    List,
    Any,
    Dict
)
import pynvml

CUDA_WITH_INDEX_TEMPLATE:str = 'cuda:{index}'

def divide_into_chunks(
    lst:List[Any], 
    k:int,
    )->List[List[Any]]:
    # Calculate the size of each chunk
    n:int = len(lst)
    k = min(k, n)
    chunk_sizes:np.ndarray = np.arange(0, n + 1, n / k).round(0).astype(int)  # Create indices for splitting
    chunk_sizes[-1] = n  # Ensure the last index matches the list length
    
    # Split the list into chunks
    chunks:List[List[Any]] = [lst[chunk_sizes[i]:chunk_sizes[i+1]] for i in range(len(chunk_sizes) - 1)]
    chunks_sum_len:int = sum(map(len, chunks))
    assert chunks_sum_len == n, f'chunks_sum_len = {chunks_sum_len} != n = {n}'
    return chunks

def get_gpu_index_2_free_memory()->Dict[int, int]:
    pynvml.nvmlInit()
    device_count:int = pynvml.nvmlDeviceGetCount()
    gpu_index_2_free_memory:Dict[int, int] = {}
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_index_2_free_memory[i] = mem_info.free
        
    pynvml.nvmlShutdown()
    return gpu_index_2_free_memory

def get_available_gpu_indices_with_free_memory(model_needed_ram:int)->List[int]:
    available_gpu:Dict[int, int] = get_gpu_index_2_free_memory()
    gpu_indices_with_free_memory:List[int] = []
    for gpu_index, free_memory in available_gpu.items():
        gpu_indices_with_free_memory.extend([gpu_index] * (free_memory // model_needed_ram))
    return gpu_indices_with_free_memory
