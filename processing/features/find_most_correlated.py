import pandas as pd
import numpy as np
from typing import (
    List, 
    Set, 
    Tuple,
)

CORRELATION_THRESHOLD_TO_DROP:float = 0.9

def features_table_2_correlated_features_pairs_list(
    X:pd.DataFrame,
    correlation_threshold:float = CORRELATION_THRESHOLD_TO_DROP,
    )->List[List[str]]:
    correlation_matrix:pd.DataFrame = X.corr()
    mask:np.ndarray = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    correlation_matrix_masked:pd.DataFrame = correlation_matrix.mask(mask).iloc[:,:]
    most_correlated_features:pd.DataFrame = correlation_matrix_masked >= correlation_threshold
    # display(most_correlated_features)
    correlated_pairs:List[List[str, str]] = []
    for col in most_correlated_features.columns:
        for row in most_correlated_features.index:
            if most_correlated_features[col][row]:
                correlated_pairs.append([col, row])
    return correlated_pairs

def pairs_list_2_sets(pairs_list:List[List[str]] )->List[Set[str]]:
    sets_list:List[Set[str]] = []
    for pair in pairs_list:
        pair_set:Set[str] = set(pair)
        if len(sets_list) == 0:
            sets_list.append(pair_set)
            continue
        sets_indices_to_union:List[int] = []
        for i, result_set in enumerate(sets_list):
            for pair_item in pair_set:
                if pair_item in result_set:
                    sets_indices_to_union.append(i)
                    break
        len_sets_indices_to_union:int = len(sets_indices_to_union)
        if len_sets_indices_to_union == 0:
            sets_list.append(pair_set)
            continue
        elif len_sets_indices_to_union == 1:
            sets_list[sets_indices_to_union[0]].update(pair_set)
        else:
            sets_list[sets_indices_to_union[0]].update(sets_list[sets_indices_to_union[1]])
            del sets_list[sets_indices_to_union[1]]
    return sets_list

def corr_table_and_most_correlated_features_sets_to_features_2_drop(
    corr_table:pd.DataFrame,
    correlated_features_sets:List[Set[str]],
    ) -> List[str]:
    features_2_drop:List[str] = []
    for correlated_features_set in correlated_features_sets:
        feature_name_2_abs_corr_sum:List[Tuple[str, float]] = []
        for feature_name in correlated_features_set:
            feature_name_2_abs_corr_sum.append(
                (
                    feature_name, corr_table[feature_name].abs().sum()
                )
            )
        feature_name_2_abs_corr_sum.sort(key=lambda x: x[1], reverse=False)
        features_2_drop.extend(list(map(lambda x: x[0], feature_name_2_abs_corr_sum[1:])))
    return features_2_drop

def most_correlated_features_to_drop(
    X:pd.DataFrame,
    correlation_threshold:float = CORRELATION_THRESHOLD_TO_DROP,
    ) -> List[str]:
    most_correlated_features_pairs_list:List[List[str]] = features_table_2_correlated_features_pairs_list(X=X, correlation_threshold=correlation_threshold)
    correlated_features_sets:List[Set[str]] = pairs_list_2_sets(pairs_list=most_correlated_features_pairs_list)
    features_2_drop:List[str] = corr_table_and_most_correlated_features_sets_to_features_2_drop(
        corr_table=X.corr(), 
        correlated_features_sets=correlated_features_sets,
    )
    return features_2_drop
