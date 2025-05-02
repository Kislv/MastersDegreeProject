from typing import (
    Any,
    List,
)
import pandas as pd
import numpy as np
import shap
from matplotlib import pyplot as plt

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
