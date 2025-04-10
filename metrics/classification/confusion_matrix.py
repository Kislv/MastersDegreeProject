import pandas as pd
from typing import (
    List,
    Optional,
    Iterable,
    Any,
    Dict,
)
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))

from configs.base import (
    DOT,
    SPACE,
)

DEFAULT_CMAP:str = 'Blues'
DEFAULT_NORMALIZE:str = 'true'
CONFUSION_MATRIX_DEFAULT_TITLE:str = 'Матрица неточностей'
CONFUSION_MATRIX_DEFAULT_XLABEL:str = 'Предсказанные значения'
CONFUSION_MATRIX_DEFAULT_YLABEL:str = 'Эталонные значения'

def plot_confusion_matrix(
    y_true:Iterable[Any],
    y_pred:Iterable[Any],
    weights:Optional[Iterable[float]] = None,
    class_names:Optional[List[str]] = None,
    normalize:Optional[str] = None,
    xlabel:Optional[str] = CONFUSION_MATRIX_DEFAULT_XLABEL,
    ylabel:Optional[str] = CONFUSION_MATRIX_DEFAULT_YLABEL,
    title:Optional[str] = CONFUSION_MATRIX_DEFAULT_TITLE,
    cmap:str=DEFAULT_CMAP,
    ):
    cm:np.ndarray = confusion_matrix(
        y_true=y_true, 
        y_pred=y_pred, 
        sample_weight=weights, 
        normalize=normalize,
    )
    disp:ConfusionMatrixDisplay = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap=cmap)
    disp.ax_.invert_yaxis()
    # plt.xlabel()
    # plt.ylabel()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_confusion_matrices(
    y_true:Iterable[Any],
    y_pred:Iterable[Any],
    weights:Optional[Iterable[float]] = None,
    class_names:Optional[List[str]] = None,
    xlabel:Optional[str] = CONFUSION_MATRIX_DEFAULT_XLABEL,
    ylabel:Optional[str] = CONFUSION_MATRIX_DEFAULT_YLABEL,
    title:Optional[str] = CONFUSION_MATRIX_DEFAULT_TITLE,
    cmap:str=DEFAULT_CMAP,
    sep:str = DOT + SPACE,
    ):

    # TODO: to config
    normalize_type_2_title:Dict[str, str] = {
        None: 'Без нормализации',
        'pred': 'Нормализация по предсказаниям',
        'true': 'Нормализация по истинным значениям',
        'all': 'Нормализация по всем значениям',
    }
    
    for normalize_type, title_postfix in normalize_type_2_title.items():
        # ConfusionMatrixDisplay
        
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            weights=weights,
            class_names=class_names,
            normalize=normalize_type,
            xlabel=xlabel,
            ylabel=ylabel,
            title=sep.join([title, title_postfix]),
            cmap=cmap,
        )

# def plot_confusion_matrix(
#     y_true: pd.Series, 
#     y_pred: pd.Series, 
#     class_names: List[str],
#     normalize:Optional[str] = DEFAULT_NORMALIZE,
#     ) -> None:
#     """
#     Plots a confusion matrix for multiclass classification results.

#     Parameters:
#     y_true (pd.Series): The true labels.
#     y_pred (pd.Series): The predicted labels.
#     class_names (List[str]): The names of the classes.

#     Returns:
#     None: Displays the confusion matrix plot.
#     """
    
#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(class_names))], normalize=normalize) # 

#     # Create a heatmap for the confusion matrix
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, cmap=DEFAULT_CMAP, xticklabels=class_names, yticklabels=class_names)
    
#     # Labels and title
#     plt.xlabel('Предсказанные значения')
#     plt.ylabel('Эталонные значения')
#     plt.title('матрица неточностей')
    
#     # Show the plot
#     plt.show()
    

def show_all_classification_metrics(
    y_true: pd.Series, 
    y_pred: pd.Series, 
    weights: Optional[Iterable[float]] = None,
    class_names: Optional[List[str]] = None,
    xlabel: Optional[str] = CONFUSION_MATRIX_DEFAULT_XLABEL,
    ylabel: Optional[str] = CONFUSION_MATRIX_DEFAULT_YLABEL,
    title: Optional[str] = CONFUSION_MATRIX_DEFAULT_TITLE,
    cmap: str = DEFAULT_CMAP,
    sep: str = DOT + SPACE
    ) -> None:
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names))
    plot_confusion_matrices(
        y_true=y_true,
        y_pred=y_pred,
        weights=weights,
        class_names=class_names,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        cmap=cmap,
        sep=sep,
    )
