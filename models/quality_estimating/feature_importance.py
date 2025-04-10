import pandas as pd
from typing import (
    Iterable,
)
import os
import sys
sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))

def normalized_feature_importance(
    feature_names:Iterable[str],
    feature_weights:Iterable[float],
    ascending:bool=False,
    )->pd.Series:
    result:pd.Series = pd.Series(
        index=feature_names, 
        data=feature_weights / (sum(feature_weights))
        ).sort_values(
            ascending=ascending,
        )
    result = result.apply(lambda x: round(x, 3))
    return result
