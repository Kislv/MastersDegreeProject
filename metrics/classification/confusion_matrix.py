import pandas as pd
from typing import (
    List,
    Optional
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt

DEFAULT_CMAP:str = 'Blues'
DEFAULT_NORMALIZE:str = 'true'

def plot_confusion_matrix(
    y_true: pd.Series, 
    y_pred: pd.Series, 
    class_names: List[str],
    normalize:Optional[str] = DEFAULT_NORMALIZE,
    ) -> None:
    """
    Plots a confusion matrix for multiclass classification results.

    Parameters:
    y_true (pd.Series): The true labels.
    y_pred (pd.Series): The predicted labels.
    class_names (List[str]): The names of the classes.

    Returns:
    None: Displays the confusion matrix plot.
    """
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(class_names))], normalize=normalize) # 

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap=DEFAULT_CMAP, xticklabels=class_names, yticklabels=class_names)
    
    # Labels and title
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Эталонные значения')
    plt.title('матрица неточностей')
    
    # Show the plot
    plt.show()
    

def show_all_classification_metrics(
    y_true: pd.Series, 
    y_pred: pd.Series, 
    class_names: List[str],
    normalize_confusion_matrix:Optional[str] = DEFAULT_NORMALIZE,
    ) -> None:
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names))
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        normalize=normalize_confusion_matrix,
    )
    