from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import sys
sys.path.append('../..')

from configs.base import (
    TAB,
)
from configs.paths import (
    DUSHA_CROWD_TRAIN_FILE_PATH,
)
from processing.text.normalization import (
    text_to_normalized_tokens,
)

RAW_CROWD_TRAIN:pd.DataFrame = pd.read_csv(
    filepath_or_buffer=DUSHA_CROWD_TRAIN_FILE_PATH, 
    sep=TAB,
)

TF_IDF_TOKENIZER:TfidfVectorizer = TfidfVectorizer(tokenizer=text_to_normalized_tokens)
TF_IDF_TOKENIZER.fit(RAW_CROWD_TRAIN.speaker_text.dropna().unique())


def tf_idf_mean(
    text:str,
    vectorizer:TfidfVectorizer = TF_IDF_TOKENIZER,
    )->float:
    tfidf_matrix = vectorizer.transform([text])
    print(f'type(tfidf_matrix) = {type(tfidf_matrix)}')

    tfidf_dense = tfidf_matrix.toarray()
    mean_tfidf_values = np.mean(tfidf_dense, axis=1)

    return mean_tfidf_values[0]
