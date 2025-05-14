import pandas as pd
import numpy as np
from typing import (
    Tuple,
    List,
    Callable,
    Optional,
)
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))

def plot_features_corr_matrix(
    X:pd.DataFrame,
    figsize:Tuple[int, int] = (18, 15),
    is_greys:bool = False,
    ):
    correlation_matrix:pd.DataFrame = X.corr()
    # correlation_matrix: pd.DataFrame = X.corr()[1:,:-1].copy()
    plt.figure(figsize=figsize)
    mask:np.ndarray = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix, 
        mask=mask,  # Mask the upper triangle
        annot=True,
        # cmap='coolwarm',
        cmap='Greys' if is_greys else 'coolwarm',  # White-black color map
        vmin=-1, vmax=1,
        center=0,
        square=True,
        fmt='.2f',
        linewidths=.5,  # Optional: adds lines between squares for clarity
        cbar_kws={"shrink": .8}  # Optional: shrinks colorbar
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_feature_class_corr_matrix(
    X_y: pd.DataFrame, 
    target_col_name: str, 
    features_renamer:Optional[Callable[[str], str]]=None,
    is_greys:bool = False,
    ):
    classes:np.ndarray = X_y[target_col_name].unique()
    features:List[str] = [col for col in X_y.columns if col != target_col_name]
    corr_matrix:pd.DataFrame = pd.DataFrame(index=features, columns=classes)
    
    for cls in classes:
        binary_target:pd.Series = (X_y[target_col_name] == cls).astype(int)
        for feature in features:
            corr:np.float64 = X_y[feature].corr(binary_target)
            corr_matrix.loc[feature, cls] = corr

    corr_matrix = corr_matrix.astype(float)
    plt.figure(figsize=(10, 10))
    corr_matrix.rename(index=features_renamer, inplace=True)
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='Greys' if is_greys else 'coolwarm',  
        vmin=-1, 
        vmax=1,
    )
    # plt.title('Correlation between features and each class')
    plt.xlabel('Класс')
    plt.ylabel('Признак')
    plt.show()
