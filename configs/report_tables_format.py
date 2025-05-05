import pandas as pd
from sklearn.metrics import (
    classification_report,
)
from pathlib import Path
from typing import (
    List,
    Dict,
    Any,
)

SKLEARN_ACCURACY_METRIC_NAME:str = 'accuracy'
SKLEARN_SUPPORT_METRIC_NAME:str = 'support'

DATAFRAME_STYLES:List[Dict[Any,Any]] = [
    {"selector": "th", "props": [("background-color", "white"), ("color", "black"), 
                                 ("font-weight", "bold"), ("text-align", "center")]},
    {"selector": "td", "props": [("background-color", "white"), ("color", "black")]},
    {"selector": "table", "props": [("border", "1px solid black"), ("border-collapse", "collapse")]}
]
TABLE_FORMAT:str = "{:.2f}"

def classification_report_formatted(
    y_true:pd.Series,
    y_pred:pd.Series,
    drop_accuracy:bool = True,
    accuracy_metric_name:str = SKLEARN_ACCURACY_METRIC_NAME,
    support_metric_name:str = SKLEARN_SUPPORT_METRIC_NAME,
    dataframe_style:List[Dict[Any,Any]] = DATAFRAME_STYLES,
    table_format:str = TABLE_FORMAT,
    )->pd.DataFrame:
    classification_report_table:pd.DataFrame = pd.DataFrame(classification_report(y_true=y_true, y_pred=y_pred, output_dict=True))
    if drop_accuracy:
        classification_report_table = classification_report_table.drop(columns=[accuracy_metric_name])
    else:
        classification_report_table = classification_report_table.drop(index=[support_metric_name])
        print(f'accuracy = {classification_report_table[accuracy_metric_name][0]}')
    styled_df:pd.DataFrame = classification_report_table.style.set_table_styles(dataframe_style).format(table_format)
    return styled_df
