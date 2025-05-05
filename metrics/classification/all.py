import pandas as pd
import numpy as np
from typing import (
    Iterable,
    Optional,
    List,
)
from sklearn.metrics import (
    classification_report,
)
import sys
import os
sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))

from configs.base import (
    DOT,
    SPACE,
)

from metrics.classification.confusion_matrix import (
    plot_confusion_matrices,
    CONFUSION_MATRIX_DEFAULT_XLABEL,
    CONFUSION_MATRIX_DEFAULT_YLABEL,
    DEFAULT_CMAP as CONFUSION_MATRIX_DEFAULT_CMAP,
)
from metrics.classification.ROC_AUC import (
    plot_roc_auc_curve,
)

def pred_proba_2_pred(pred_proba:pd.DataFrame) -> pd.Series:
    return pd.Series(pred_proba.values.argmax(axis=1), index=pred_proba.index).apply(lambda x: pred_proba.columns[x])

def macro_accuracy(y_true:pd.Series, y_pred:pd.Series) -> float:
    metric_vals:List[float] = []
    for class_name, group in y_true.groupby(by=y_true):
        metric_val:float = accuracy_score(group, y_pred[group.index])
        metric_vals.append(metric_val)
        print(class_name, metric_val)
    return np.mean(metric_vals)

def show_all_classification_metrics(
    y_true: pd.Series, 
    y_pred: Optional[pd.Series] = None, 
    y_pred_proba: Optional[pd.DataFrame] = None, 
    weights: Optional[Iterable[float]] = None,
    class_names: Optional[List[str]] = None,
    plot_cm:bool=True,
    plot_roc_auc:bool=False,
    xlabel: Optional[str] = CONFUSION_MATRIX_DEFAULT_XLABEL,
    ylabel: Optional[str] = CONFUSION_MATRIX_DEFAULT_YLABEL,
    title: Optional[str] = None, # CONFUSION_MATRIX_DEFAULT_TITLE
    cmap: str = CONFUSION_MATRIX_DEFAULT_CMAP,
    sep: str = DOT + SPACE
    ) -> None:
    if y_pred is None and y_pred_proba is None:
        raise ValueError("y_pred or y_pred_proba must be provided")
    if y_pred is None:
        y_pred = pred_proba_2_pred(y_pred_proba)    
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names))
    if plot_cm:
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
    if plot_roc_auc:
        if y_pred_proba is not None:
            plot_roc_auc_curve(
                test=y_true,
                pred_proba=y_pred_proba,
            )
        else:
            raise ValueError("y_pred_proba must be provided")
