import numpy as np
from dataclasses import (
    dataclass,
)
import wave
from pathlib import Path
import pyloudnorm as pyln
from typing import (
    Callable,
)
import torch

from configs.base import (
    RB_FILE_READING_MODE,
)

from models.asr.whisper import (
    # whisper_audio_file_2_transcription,
    whisper_tensor_with_sr_transcription,
)

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

@dataclass 
class Audio:
    sample_width:int
    sr:int
    n_frames:int
    data:np.ndarray
    n_channels:int=1
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
        path:Path,
        reading_mode:str = RB_FILE_READING_MODE,
        ):
        with wave.open(str(path), reading_mode) as wav_file:
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
                n_channels=n_channels,
                sample_width=sample_width,
                sr=frame_rate,
                n_frames=n_frames,
                data=signal_array,
            )
            return audio
    def new_data_copy(
        self,
        data:np.ndarray
        ):
        return Audio(
            sample_width=self.sample_width,
            sr=self.sr,
            n_frames=self.n_frames,
            n_channels=self.n_channels,
            data=data
        )
    
    def sample_dtype(
        self,
        )->type:
        return type(self).sample_width_2_dtype(sample_width=self.sample_width)
    
    def volume(
        self,
        data_type:type = np.float64,
        )->np.float64:
        meter:pyln.meter.Meter = pyln.Meter(self.sr)
        return meter.integrated_loudness(self.data.astype(data_type))
    
    def transcribe(
        self,
        transcriber:Callable[
            [
            torch.Tensor, 
            int, 
            WhisperProcessor, 
            WhisperForConditionalGeneration
            ], 
            str
        ] = whisper_tensor_with_sr_transcription
        )->str:
        return transcriber(
            tensor=torch.Tensor(self.data.copy()), 
            sr=self.sr,
        )

