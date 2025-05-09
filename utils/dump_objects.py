from typing import (
    Any,
    
)
from configs.base import (
    WB_OPEN_FILE_MODE,
    RB_OPEN_FILE_MODE,
)
from pathlib import Path
import pickle

def dump_2_file(
    object:Any, 
    path:Path,
    open_file_mode:str = WB_OPEN_FILE_MODE,
    )->None:
    
    with open(path, open_file_mode) as f:
        pickle.dump(object, f)

def read_dumped(
    path:Path,
    open_file_mode:str = RB_OPEN_FILE_MODE,
    )->Any:
    with open(path, open_file_mode) as f:
        obj = pickle.load(f)
        return obj
