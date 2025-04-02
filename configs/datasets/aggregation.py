import pandas as pd
from typing import (
    Optional,
)
from configs.base import (
    AGGREGATED_KEYWORD,
)

import sys

sys.path.append('../..')

AGG_THRESHOLD:float = 0.7
# Group by hash_id and calculate the mode for annotator_emo
# def aggregate_by_mode(
#     group:pd.Series, 
#     # agg_col_name:str,
#     aggregated_keyword:str = AGGREGATED_KEYWORD,
#     agg_threshold:float = AGG_THRESHOLD,
#     )->Optional[pd.Series]:
#     # Calculate the mode
#     mode_value = group.mode().iloc[0]
    
#     # Calculate the frequency of the mode
#     mode_count = (group == mode_value).sum()
    
#     # Check if frequency of mode is >= 50% of the group size
#     if (mode_count/len(group)) >= agg_threshold :
#         group[aggregated_keyword] = mode_value
#         return group
#     else:
#         return None
    

def aggregate_by_mode(
    group: pd.Series,
    agg_threshold: float = AGG_THRESHOLD,
) -> Optional[pd.Series]:
    # Calculate the mode
    mode_value = group.mode().iloc[0]
    
    # Calculate the frequency of the mode
    mode_count = (group == mode_value).sum()
    
    # Check if frequency of mode is >= threshold
    if (mode_count / len(group)) >= agg_threshold:
        return mode_value
    else:
        return None