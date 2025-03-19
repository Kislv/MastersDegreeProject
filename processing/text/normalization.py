import string
from typing import (
    Set,
    Iterable,
    Callable,
    List
)
import pymorphy3
import nltk
import sys
sys.path.append('../..')

from configs.base import (
    SPACE,
)

PUNCTUATION_SYMBOLS:Set[str] = set(string.punctuation)

MORPH:pymorphy3.analyzer.MorphAnalyzer = pymorphy3.MorphAnalyzer()
LEMMATIZER:Callable = lambda x: MORPH.parse(x)[0].normal_form
TRANSFORMATIONS = [str.lower, LEMMATIZER]

TOKENIZER_LANGUAGE:str = 'russian'
TOKENIZER:Callable = lambda x: nltk.word_tokenize(x, language=TOKENIZER_LANGUAGE)

def tokens_normalization(
        transformations:Iterable[Callable], 
        tokens: Iterable[str], 
        delete_punctuation:bool=False,
        punctuation_symbols:Set[str] = PUNCTUATION_SYMBOLS,
    ) -> Iterable[str]:
    
    for transformation in transformations:
        tokens = map(transformation, tokens)
    
    if delete_punctuation:
        # tokens = map(lambda x: x.translate(punct_to_none), tokens)
        tokens = filter(lambda x: x not in punctuation_symbols, tokens)
    return tokens

def text_to_normalized_tokens(
        text: str, 
        delete_punctuation:bool=False,
        tokenizer:Callable = TOKENIZER,
        punctuation_symbols:Set[str] = PUNCTUATION_SYMBOLS,
        ) -> Iterable[str]:
    tokens:List[str] = tokenizer(text)
    normalized_tokens:Iterable[str] = tokens_normalization(
        transformations=TRANSFORMATIONS,
        tokens=tokens,  
        delete_punctuation=delete_punctuation,
        punctuation_symbols=punctuation_symbols,
    )
    
    return normalized_tokens

def normalized_tokens_2_normalized_text(
    tokens:Iterable[str],
    sep:str = SPACE
    )->str:
    return sep.join(tokens)

def text_2_normalized_text(
    text:str,
    delete_punctuation:bool=False,
    tokenizer:Callable = TOKENIZER,
    punctuation_symbols:Set[str] = PUNCTUATION_SYMBOLS,
    words_sep:str = SPACE,
    ):

    normed_tokens:Iterable[str] = text_to_normalized_tokens(
        text=text,
        delete_punctuation=delete_punctuation,
        tokenizer=tokenizer,
        punctuation_symbols=punctuation_symbols,
        )
    return normalized_tokens_2_normalized_text(
        tokens=normed_tokens,
        sep=words_sep,
    ) 