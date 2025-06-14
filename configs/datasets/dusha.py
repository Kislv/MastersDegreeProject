from enum import Enum
from typing import Callable
from functools import partial
import sys
import pandas as pd

sys.path.append('../..')
from configs.base import (
    AGGREGATED_KEYWORD,
    DEFAULT_SEP,
)
from configs.datasets.aggregation import (
    aggregate_by_mode,
)

TEXT_COLUMN_NAME:str = 'speaker_text'
SPEAKER_EMOTION_FIELD_NAME:str = 'speaker_emo'
ANNOTATOR_EMOION_FIELD_NAME:str = 'annotator_emo'
ANNOTATOR_AGGREGATED_FIELD_NAME:str = DEFAULT_SEP.join(
    [
        ANNOTATOR_EMOION_FIELD_NAME, 
        AGGREGATED_KEYWORD,
    ]
)
TEXT_2_EMOTION_TARGET_FIELD_NAME:str = SPEAKER_EMOTION_FIELD_NAME
DURATION_COLUMN_NAME:str = 'duration'
EMOTION_COLUMN_NAME:str = 'annotator_emo'
HASH_ID_COLUMN_NAME:str = 'hash_id'
SAMPLE_RATE:int = 16_000

# TODO: ensure it is correct
class GoldenEmo(Enum):
    # angry = 1
    # sad = 2
    # neutral = 3
    # positive = 4
    # other = 5

#  habr
    positive = 1
    sad = 2
    angry = 3
    neutral = 4
    other = 5

ANNOTATOR_ANSWERS_AGGREGATING_THRESHOLD:float = 0.7

def aggregate_crowd(
    df:pd.DataFrame,
    agg_func:Callable = aggregate_by_mode,
    agg_col_name:str=ANNOTATOR_EMOION_FIELD_NAME,
    by_col_name:str=HASH_ID_COLUMN_NAME,
    aggregated_col_name:str=ANNOTATOR_AGGREGATED_FIELD_NAME,
    )->pd.Series:
    hash_id_2_annotator_emotion:pd.Series = pd.Series(index=df[by_col_name], data=df[agg_col_name])
    agg_func:Callable = partial(agg_func, agg_col_name=agg_col_name, aggregated_keyword=aggregated_col_name)
    # TODO: Optimize using sort and pd.Series
    # groupped_by_df:pd.DataFrame = df.groupby(by=by_col_name).apply(agg_fun
    
    return hash_id_2_annotator_emotion.groupby(hash_id_2_annotator_emotion.index).apply(aggregate_by_mode)
