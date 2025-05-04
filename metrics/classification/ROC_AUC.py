import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    auc,
)
from sklearn.preprocessing import (
    label_binarize,
)

import sys
import os
sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))

from typing import (
    List,
)
def plot_roc_auc_curve(
    test:pd.Series,
    pred_proba:pd.DataFrame,
    ):
    class_names:List[str] = pred_proba.columns.tolist()
    y_test_bin:np.ndarray = label_binarize(test, classes=class_names)

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], pred_proba[class_name])
        AUC:np.float64 = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {AUC:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Мультикслассовая ROC-AUC')
    plt.legend(loc="lower right")
    plt.show()
