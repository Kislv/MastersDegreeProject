import numpy as np
from dataclasses import (
    dataclass,
)
import wave
from pathlib import Path
import pyloudnorm as pyln
from typing import (
    Callable,
    Optional,
    List,
    Set,
    Union,
)
import torch
import librosa
import copy

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

from configs.base import (
    RB_OPEN_FILE_MODE,
    SECONDS_QUANTITY_IN_MINUTE,
    BREAK_LINE,
    RUSSIAN_VOWELS,
    SPACE,
    EMPTY,
)

from processing.text.normalization import (
    TOKENIZER,
    PUNCTUATION_SYMBOLS,
)

@dataclass 
class WAVFilePathInitArgs:
    path:Path
    transcription:str
    reading_mode:str = RB_OPEN_FILE_MODE

@dataclass 
class Audio:
    hash: str
    sample_width:int
    sr:int
    n_frames:int
    data:np.ndarray
    _transcription:Union[str, float]
    n_channels:int=1
    # TODO: Try to use "__"

    @classmethod
    def sample_width_2_dtype(
        cls,
        sample_width:int,
        )->type:
        if sample_width == 1:
            return np.uint8
        elif sample_width == 2:
            return np.int16
        elif sample_width == 4:
            return np.int32
        else:
            raise ValueError("Unsupported sample width")

    @classmethod
    def wav_file_path_init(
        cls,
        # path:Path,
        # transcription:Optional[str] = None,
        # reading_mode:str = RB_OPEN_FILE_MODE,
        arguments:WAVFilePathInitArgs
        ):
        if not arguments.path.exists():
            raise FileNotFoundError
        with wave.open(str(arguments.path), arguments.reading_mode) as wav_file:
            n_channels:int = wav_file.getnchannels()
            frame_rate:int = wav_file.getframerate()
            sample_width:int = wav_file.getsampwidth()
            n_frames:int = wav_file.getnframes()
            signal:bytes = wav_file.readframes(n_frames)
            dtype:type = cls.sample_width_2_dtype(sample_width=sample_width)
            signal_array = np.frombuffer(
                signal, 
                dtype=dtype,
            )
            audio:Audio = Audio(
                hash=arguments.path.stem,
                n_channels=n_channels,
                sample_width=sample_width,
                sr=frame_rate,
                n_frames=n_frames,
                data=signal_array,
                _transcription=arguments.transcription,
            )
            return audio
    # [''] understand what does it mean


    def transcription(
        self,

        )->List[str]:
        if isinstance(self._transcription, float):
            self._transcription = EMPTY
        return self._transcription 

    def joined_transcription(
        self,
        sep:str = BREAK_LINE,
        )->str:
        return sep.join(
            self._transcription
        )

    def new_data_copy(
        self,
        data:np.ndarray
        ):
        self_copy:Audio = copy.deepcopy(self)
        self_copy.data=data
        return self_copy
    
    def sample_dtype(
        self,
        )->type:
        return type(self).sample_width_2_dtype(sample_width=self.sample_width)
