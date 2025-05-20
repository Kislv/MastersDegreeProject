import numpy as np
from typing import (
    List,
    Any,
)

CUDA_WITH_INDEX_TEMPLATE:str = 'cuda:{index}'

def divide_into_chunks(
    lst:List[Any], 
    k:int,
    )->List[List[Any]]:
    # Calculate the size of each chunk
    n:int = len(lst)
    chunk_sizes:np.ndarray = np.arange(0, n + 1, n / k).round(0).astype(int)  # Create indices for splitting
    chunk_sizes[-1] = n  # Ensure the last index matches the list length
    
    # Split the list into chunks
    chunks:List[List[Any]] = [lst[chunk_sizes[i]:chunk_sizes[i+1]] for i in range(len(chunk_sizes) - 1)]
    return chunks

