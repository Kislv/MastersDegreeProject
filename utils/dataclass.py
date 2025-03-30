import sys
from typing import (
    Dict,
    Any,
)
sys.path.append('..')
from configs.base import (
    DEFAULT_SEP,
)

def add_key_prefix_inplace(prefix:str, d:Dict[str, Any], sep:str = DEFAULT_SEP):
    keys = list(d.keys())

    for k in keys:
        d[prefix + sep + k] = d.pop(k)
    return d

def flatten_dict(d:Dict[str,Any], name_sep:str = DEFAULT_SEP)->Dict[str, Any]:
    flattened_dict:Dict[str, Any] = {}
    # TODO to dome config
    keys_collision_message:str = 'key already in flattened_dict!'
    for key, value in d.items():
        if not isinstance(value, Dict):
            if key in flattened_dict:
                print(keys_collision_message)
            flattened_dict[key] = value
        else:
            flattened_value:Dict[str,Any] = add_key_prefix_inplace(prefix=key, d=value.copy())
            if len(set(flattened_value.keys()).intersection(set(flattened_dict.keys()))) > 0:
                print(keys_collision_message)
            
            flattened_dict.update(flattened_value)
    return flattened_dict
