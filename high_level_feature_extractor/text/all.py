from dataclasses import dataclass
import sys
import numpy as np
from typing import (
    List,
    Callable,
    Set,
    Optional,
)
from high_level_feature_extractor.text.profanity import (
    text_2_is_contain_swear_words,
    PROFANITY_WORD_FILTER_LANG_NAME,
)

sys.path.append('../..')
from configs.base import(
    SPACE,
    EMPTY,
)

from processing.text.normalization import (
    text_2_normalized_text,
    TOKENIZER,
    PUNCTUATION_SYMBOLS,
)

from high_level_feature_extractor.text.meaning import (
    tf_idf_mean,
)
from high_level_feature_extractor.text.POS import (
    POS_ratio as POS_ratio_class,
)

@dataclass
class TranscriptionHighLevelFeatures:
    mean_words_length:float
    # profanity_words_quantity:int
    profanity_words_ratio:float
    # TODO preprocess, delete Nones
    meaning:Optional[float] = None
    POS_ratio:Optional[POS_ratio_class] = None
    @classmethod
    def text_init(
        cls,
        text:str,
        tokenizer:Callable = TOKENIZER,
        punctuation_symbols:Set[str] = PUNCTUATION_SYMBOLS,
        words_sep:str = SPACE,
        ):
        letters_and_seps_only:str = EMPTY.join(
            list(
                filter(
                    lambda letter: letter.isalpha() or letter == words_sep, 
                    text,
                )
            )
        )
        normalized_text:str = text_2_normalized_text(
            text=letters_and_seps_only,
            delete_punctuation=True,
            tokenizer=tokenizer,
            punctuation_symbols=punctuation_symbols,
            words_sep=words_sep,
        )
    
        words:List[str] = letters_and_seps_only.split(sep=words_sep)
        words_quantity:int = len(words)
        return TranscriptionHighLevelFeatures(
            
            mean_words_length = (
                sum(
                    map(lambda word: len(word), words)
                ) / words_quantity
             ) if words_quantity > 0 else np.nan,

            profanity_words_ratio = (
                sum(
                    map(
                        lambda word: text_2_is_contain_swear_words(word), normalized_text.split(words_sep)
                    )
                ) / words_quantity
            ) if words_quantity > 0 else np.nan,
            meaning=tf_idf_mean(
                text=normalized_text,
            ),
            POS_ratio=POS_ratio_class.text_init(
                letters_and_seps_only_text=letters_and_seps_only,
            )
        )
