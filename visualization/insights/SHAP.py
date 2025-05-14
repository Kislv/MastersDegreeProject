from typing import (
    Any,
    List,
    Optional,
    Iterable,
    Union,
)
import pandas as pd
import numpy as np
import shap
from matplotlib import pyplot as plt
import matplotlib


def shap_tree_feature_importance(
    model:Any,
    X_test:pd.DataFrame,
    y_test:pd.Series,
    column_names:List[str],
    class_names:List[str],
    ):
    explainer:shap.explainers._tree.TreeExplainer = shap.TreeExplainer(model)
    shap_values:np.ndarray = explainer.shap_values(X_test, y=y_test)

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values, 
        X_test,
        feature_names=column_names,
        plot_type="bar",
        max_display=20,
        plot_size=(12, 8),
        show=False,
        class_names=class_names,
    )

    plt.gcf().set_size_inches(12, 8)  # Set figure size
    plt.xlabel(
        'mean(|SHAP value|) - вклад признака', 
        fontsize=12,
    )
    plt.ylabel('Признаки', fontsize=12)
    plt.title('Важность признаков, основанная на значениях Шепли', fontsize=14)
    plt.yticks(fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
# index = df.columns.get_loc('B')
def feature_shap_dependence_plot(
    tree_model:Any,
    X:pd.DataFrame,
    y:pd.Series,
    feature_name:str, # mean_volumne
    feature_interaction_name:str,
    cols_names:Optional[Iterable[str]] = None,
    ):
    feature_index:int = X.columns.get_loc(feature_name)
    feature_interaction_index:int = X.columns.get_loc(feature_interaction_name)
    explainer:shap.explainers._tree.TreeExplainer = shap.TreeExplainer(tree_model)
    shap_values:np.ndarray = explainer.shap_values(
        X=X,
        y=y,
    )
    n_classes:int = len(tree_model.classes_)
    _, axes = plt.subplots(nrows=n_classes, ncols=1, figsize=(10, 5*n_classes))

    if cols_names is None:
        cols_names = X.columns
    
    for class_idx in range(n_classes):
        plt.sca(axes[class_idx])  # Set the current axis
        shap_values_for_class = shap_values[:, :, class_idx] 
        ax:matplotlib.axes._axes.Axes = axes[class_idx]
        shap.dependence_plot(
            feature_index, 
            shap_values_for_class,
            X,
            feature_names=cols_names,
            title=tree_model.classes_[class_idx],
            interaction_index=feature_interaction_index,
            ax=ax,
            show=False,
        )
        ax.figure.set_size_inches(12, 15)
        if class_idx == n_classes//2:
            ax.set_ylabel('Значение Шепли')
        else:
            ax.set_ylabel(f'')
        if class_idx == n_classes-1:
            ax.set_xlabel(cols_names[feature_index])
        else: 
            ax.set_xlabel('')
    plt.tight_layout()
    plt.show()

def shap_row_waterfall_plot(
    tree_model:Any,
    row_index:Any,
    X:pd.DataFrame,
    y:pd.Series,
    cols_names:Optional[Iterable[str]] = None,
    # top_important_k:int = 10 # TODO: extract k most important, visualize them, other features sum up and name f'{m} other features'
    ):

    true_label:str = y[row_index]
    explainer = shap.TreeExplainer(model=tree_model)
    shap_values = explainer.shap_values(X[X.index == row_index])

    class_idx:int = pd.Index(tree_model.classes_).get_loc(true_label) 

    shap_values_row = shap_values[0]
    class_name = tree_model.classes_[class_idx]

    plt.figure(figsize=(10, 6))

    plt.title(f'Значения Шепли для конкретной записи. Предсказанный класс - {class_name}, истинный класс - {true_label}')
    plt.tight_layout()

    shap.plots.waterfall(
        shap.Explanation(
            # values=shap_values_row[:top_important_k,0], # not sorted by importance
            values=shap_values_row[:,0],
            base_values=explainer.expected_value[class_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            data=X.loc[row_index],
            feature_names=cols_names,
        ),
        # color=plt.cm.Greys,  # <-- here is the change to greyscale
    )
