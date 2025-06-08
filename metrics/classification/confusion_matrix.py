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
    roc_curve, 
    auc,
)
from sklearn.preprocessing import (
    label_binarize,
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
    if class_names is None:
        class_names = list(set(y_true).union(set(y_pred)))
    cm:np.ndarray = confusion_matrix(
        y_true=y_true, 
        y_pred=y_pred, 
        labels=class_names,
        sample_weight=weights, 
        normalize=normalize,
    )
    disp:ConfusionMatrixDisplay = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap=cmap)
    disp.ax_.invert_yaxis()
    plt.grid(False)
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
    title:Optional[str] = None, # CONFUSION_MATRIX_DEFAULT_TITLE
    cmap:str=DEFAULT_CMAP,
    sep:str = DOT + SPACE,
    ):
    if class_names is None:
        class_names = sorted(list(set(y_true).union(set(y_pred))))

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
            title=sep.join(filter(lambda x: x is not None, [title, title_postfix])),
            cmap=cmap,
        )


