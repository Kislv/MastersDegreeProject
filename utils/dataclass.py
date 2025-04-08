import sys
from typing import (
    Dict,
    Any,
    Optional
)
sys.path.append('..')
from configs.base import (
    DEFAULT_SEP,
)

def add_key_prefix_inplace(
    prefix:Optional[str], 
    d:Dict[str, Any], 
    sep:str = DEFAULT_SEP,
    ):
    keys = list(d.keys())
    for k in keys:
        key:str = k
        if prefix is not None:
            key = prefix + sep + key
        if key in d.keys():
            print(f'key {key} already in dict!')
        d[key] = d.pop(k)
    return d

def flatten_dict(
    d:Dict[str, Any], 
    name_sep:str = DEFAULT_SEP,
    add_prefix:bool = True,
    )->Dict[str, Any]:
    result:Dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, Dict):
            flattened_value:Dict[str, Any] = flatten_dict(
                d=value,
                name_sep=name_sep,
                add_prefix=add_prefix,
            )
            if add_prefix:
                add_key_prefix_inplace(
                    prefix=key, 
                    d=flattened_value, 
                    sep=name_sep,
                )
            for value_key, value_value in flattened_value.items():
                if value_key in result.keys():
                    print(f'key {value_key} already in flattened_dict!')
                result[value_key] = value_value
            
        else:
            if key in result.keys():
                print(f'key {key} already in flattened_dict!')
            result[key] = value
            
    return result
