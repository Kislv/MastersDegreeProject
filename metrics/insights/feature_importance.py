from typing import (
    Iterable,
    Tuple,
    Optional,
)
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))

from configs.plots import (
    FEATURE_RU,
    NORMED_WEIGHT_RU,
    XTICKS_DEFAULT_ROTATION,
)
from models.quality_estimating.feature_importance import (
    normalized_feature_importance,
)

NORMED_FEATURE_IMPORTANCE_DEFAULT_FIGSIZE:Tuple[int, int] = (8, 6)

def normalized_feature_importance_plot(
    feature_names:Iterable[str],
    feature_weights:Iterable[float],
    ascending:bool=False,
    feature_axis_label:str = FEATURE_RU,
    weight_axis_label:str = NORMED_WEIGHT_RU,
    xticks_rotation:int = XTICKS_DEFAULT_ROTATION,
    figsize:Tuple[int, int] = NORMED_FEATURE_IMPORTANCE_DEFAULT_FIGSIZE,
    top_n:Optional[int]=None,
    )->None:
    # normed_feature_importance:pd.Series = normalized_feature_importance(
    #     feature_names=feature_names,
    #     feature_weights=feature_weights,
    #     ascending=ascending,
    # )
    # plt.figure(figsize=figsize)  # 12-inch width, 6-inch height
    # if top_n is not None:
    #     normed_feature_importance = normed_feature_importance.head(top_n)
    #     plt.title(label=f'топ-{top_n}')
    # df:pd.DataFrame = normed_feature_importance.reset_index()
    # df.columns = [feature_axis_label, weight_axis_label]
    # sns.barplot(x=feature_axis_label, y=weight_axis_label, data=df)
    # plt.xticks(rotation=xticks_rotation)
    # plt.show()


    normed_feature_importance: pd.Series = normalized_feature_importance(
        feature_names=feature_names,
        feature_weights=feature_weights,
        ascending=ascending,
    )
    plt.figure(figsize=figsize)
    
    if top_n is not None:
        normed_feature_importance = normed_feature_importance.head(top_n)
        plt.title(label=f'топ-{top_n}')
    
    # Prepare horizontal bar plot data
    features = normed_feature_importance.index
    weights = normed_feature_importance.values
    
    plt.barh(y=features, width=weights)
    plt.xlabel(weight_axis_label)
    plt.ylabel(feature_axis_label)
    
    # Adjust y-axis for proper orientation (highest weight at top)
    plt.gca().invert_yaxis()
    
    # Optional: Rotate x-axis labels if needed (though less critical for horizontal bars)
    plt.xticks(rotation=xticks_rotation)
    
    plt.show()