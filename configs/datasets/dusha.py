from enum import Enum

TEXT_COLUMN_NAME:str = 'speaker_text'
SPEAKER_EMOTION_FIELD_NAME:str = 'speaker_emo'
TEXT_2_EMOTION_TARGET_FIELD_NAME:str = SPEAKER_EMOTION_FIELD_NAME
DURATION_COLUMN_NAME:str = 'duration'
EMOTION_COLUMN_NAME:str = 'annotator_emo'
HASH_ID_COLUMN_NAME:str = 'hash_id'

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
