from typing import (
    Set,
)

EMPTY:str = ''
RB_OPEN_FILE_MODE:str = 'rb'
WB_OPEN_FILE_MODE:str = 'wb'

SPACE:str = ' '
DOT:str = '.'
COLON:str = ':'
SECONDS_QUANTITY_IN_MINUTE:int = 60
BREAK_LINE:str = '\n'
TAB:str = '\t'
DEFAULT_SEP:str = '_'

# extensions
CSV:str = 'csv'
DOT_CSV:str = DOT + CSV
JSONL:str = 'jsonl'
DOT_JSONL:str = DOT + JSONL
PT:str = 'pt'

DROP_DUPLICATES_KEEP_FIRST:str = 'first'

EPSILON:float = 1e-6

RUSSIAN_VOWELS:Set[str] = {'а', 'е', 'ё', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я'}

JOIN_HOW_INNER:str = 'inner'

AGGREGATED_KEYWORD:str = 'aggregated'
