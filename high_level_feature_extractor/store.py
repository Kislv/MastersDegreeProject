from pathlib import Path
from typing import List, Optional
import os
import sys
import pandas as pd
sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))

from high_level_feature_extractor.extractor import (
    HashHLF
)
from configs.paths import (
    PROCESSED_DUSHA_CROWD_TRAIN_HLF_STABLE_VERSION_FILE_PATH,
)
from configs.datasets.dusha import (
    ANNOTATOR_AGGREGATED_FIELD_NAME,
    aggregate_crowd,
)

# def read_HLF_file(
#     HLF_file_path:Path = PROCESSED_DUSHA_CROWD_TRAIN_HLF_STABLE_VERSION_FILE_PATH,
#     )->List[HashHLF]:
#     hash_HLF_list:List[HashHLF] = []
#     with open(HLF_file_path) as f:
#         for line in f:
#             el:Optional[HashHLF] = eval(eval(line)) if eval(line) is not None else None
#             if el is not None:
#                 hash_HLF_list.append(el)
                
#     return hash_HLF_list

# def raw_crowd_2_raw_crowd_HLF_table_format(
#     raw_crowd:pd.DataFrame,
#     agg_col_name:str = ANNOTATOR_AGGREGATED_FIELD_NAME,
#     ):
#     raw_crowd_test_agged:pd.DataFrame = aggregate_crowd(
#         df=raw_crowd, 
#         aggregated_col_name=agg_col_name,
#     )

#     raw_crowd_unique_hashes:pd.DataFrame = raw_crowd_test_agged[~raw_crowd_test_agged.index.duplicated()]
#     return raw_crowd_unique_hashes

# def HLF_withspeaker_emottions_table(
#     raw_crowd:pd.DataFrame,
#     HLF_file_path:Path,
#     agg_col_name:str = ANNOTATOR_AGGREGATED_FIELD_NAME,
#     filter_agg_emo_equal_2_speaker_emo:bool=False,
#     )->pd.DataFrame:
#     # hash_HLF_list:List[HashHLF] = read_HLF_file(HLF_file_path=HLF_file_path)
#     # HLF_table:pd.DataFrame = hash_HLF_list_2_df(l=hash_HLF_list)

#     # raw_crowd_unique_hashes_with_speaker_emo_with_speaker_text:pd.DataFrame = raw_crowd_2_raw_crowd_HLF_table_format(raw_crowd=raw_crowd, agg_col_name=agg_col_name)
#     hash_id_list:List[str] = raw_crowd.hash_id.to_list()
#     hash_id_2_annotator_emo:pd.Series = pd.Series(
#         index=hash_id_list, 
#         data=raw_crowd.annotator_emo.to_list(),
#     ).sort_index()
#     hash_2_aggregated_target:pd.Series = hash_id_2_annotator_emo.groupby(hash_id_2_annotator_emo.index).apply(aggregate_by_mode).dropna()
#     hash_2_aggregated_target.name = agg_col_name

#     if filter_agg_emo_equal_2_speaker_emo:
#         print(f'Before comparing with speaker_emo hash_2_aggregated_target.shape[0]= {hash_2_aggregated_target.shape[0]}')
#         hash_id_2_speaker_emo:pd.Series = pd.Series(
#             index = hash_id_list,
#             data = raw_crowd.speaker_emo.to_list(),
#             name=raw_crowd.speaker_emo.name,
#         )
#         hash_id_2_speaker_emo = hash_id_2_speaker_emo[~hash_id_2_speaker_emo.index.duplicated(keep='first')]
#         hash_2_aggregated_target_with_speaker_emo:pd.DataFrame = pd.concat([hash_2_aggregated_target, hash_id_2_speaker_emo], axis=1)
#         hash_2_aggregated_target = hash_2_aggregated_target_with_speaker_emo[hash_2_aggregated_target_with_speaker_emo.speaker_emo == hash_2_aggregated_target_with_speaker_emo[agg_col_name]][agg_col_name]
#         print(f'Before comparing with speaker_emo hash_2_aggregated_target.shape[0]= {hash_2_aggregated_target.shape[0]}')


#     HLF_with_speaker_emotions:pd.DataFrame = HLF_tablabel_encoderjoin(
#         hash_2_aggregated_target, 
#         how=JOIN_HOW_INNER,
#     )
#     return HLF_with_speaker_emotions