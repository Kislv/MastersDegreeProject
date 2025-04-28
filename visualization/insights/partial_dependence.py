from typing import (
    Any,
    List,
)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.utils._bunch import Bunch
import os
import sys
from catboost import CatBoost, CatBoostClassifier
sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))

from configs.plots import (
    PARTIAL_DEPENDENCE_KIND_AVERAGE,
    PARTIAL_DEPENDENCE_GRID_VALUES,
)
from high_level_feature_extractor.extractor import (
    HLF_ENG_NAME_2_RU_NAME,
)

def cb_model_2_feature_importance_series(
    model:CatBoostClassifier,
    )->pd.Series:
    return pd.Series(index=model.feature_names_, data=model.feature_importances_).sort_values(ascending=False)


def plot_partial_dependence(
    model:Any,
    X:pd.DataFrame,
    model_class_names: List[str],
    feature_names: List[str],
    suptitle:str = 'Частичная зависимость',
    grid_values_str:str=PARTIAL_DEPENDENCE_GRID_VALUES,
    kind:str=PARTIAL_DEPENDENCE_KIND_AVERAGE,
    ):
    features_quantity: int = len(feature_names)
    plt.figure(figsize=(6, 9))  # Adjusted figure size since we'll have fewer subplots
    plt.suptitle(
        suptitle, 
        fontsize=16, 
        y=1.02,
    )

    for feature_idx, feature in enumerate(feature_names):
        idx:int = X.columns.get_loc(feature)
        ax = plt.subplot(features_quantity, 1, feature_idx + 1)
        
        pd_result:Bunch = partial_dependence(
            model,
            X,
            features=[idx],
            kind=PARTIAL_DEPENDENCE_KIND_AVERAGE,
            grid_resolution=20,
        )
        for class_idx, class_name in enumerate(model_class_names):
            ax.plot(
                pd_result[grid_values_str][0],
                pd_result[kind][class_idx],
                label=class_name
            )
        
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{HLF_ENG_NAME_2_RU_NAME[feature]}', fontsize=16)
        ax.legend()  # Add legend to show which line corresponds to which class

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.show()

def plot_partial_dependence_catboost(
    model:CatBoost,
    X:pd.DataFrame,
    top_features: int = 3,
    suptitle:str = 'Частичная зависимость',
    grid_values_str:str=PARTIAL_DEPENDENCE_GRID_VALUES,
    kind:str=PARTIAL_DEPENDENCE_KIND_AVERAGE,
    ):
    plot_partial_dependence(
        model=model,
        X=X,
        model_class_names=model.classes_,
        feature_names=cb_model_2_feature_importance_series(model=model).head(top_features).index.to_list(),
        suptitle=suptitle,
        grid_values_str=grid_values_str,
        kind=kind,
    )
