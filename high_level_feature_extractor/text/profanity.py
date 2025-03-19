from bdw.check import Check # https://github.com/FlacSy/BadWords/tree/master
from processing.text.normalization import (
    text_2_normalized_text,
)

PROFANITY_WORD_FILTER_LANG_NAME:str = 'ru'

def normalized_text_2_is_contain_swear_words(
    normalized_text:str,
    lang:str = PROFANITY_WORD_FILTER_LANG_NAME,
    ):
    filter:bdw.check.Check = Check(languages=[lang])
    return filter.filter_profanity(normalized_text, language=lang)
    

def text_2_is_contain_swear_words(
    text:str,
    lang:str = PROFANITY_WORD_FILTER_LANG_NAME,
    ):
    normalized_text:str = text_2_normalized_text(text=text)
    return normalized_text_2_is_contain_swear_words(
        normalized_text=normalized_text,
        lang=lang,
    )
