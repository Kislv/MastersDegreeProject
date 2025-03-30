from dataclasses import (
    dataclass,
)
import numpy as np
from pathlib import Path
import pyloudnorm as pyln
from typing import (
    Callable,
    Optional, 
    Set,
)
import sys
from scipy.io import wavfile
from audio import Audio
from scipy.fft import rfft, irfft
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
import librosa
sys.path.append('..')

from models.asr.whisper import (
    whisper_audio_file_2_transcription,
)
from volume.human_speech import (
    HIGH_FREQUENCY_SPEECH_THRESHOLD,
)
from volume.human_speech import (
    HUMAN_SPEECH_FREQ_BOTTOM,
    HUMAN_SPEECH_FREQ_TOP,
    HIGH_FREQUENCY_SPEECH_THRESHOLD,
)

from high_level_feature_extractor.text.all import (
    TranscriptionHighLevelFeatures,
)
from models.asr.whisper import (
    whisper_tensor_with_sr_transcription
)

from configs.base import (
    RUSSIAN_VOWELS,
    EPSILON,
)

@dataclass
class PronounceSpeed:
    WPS:int
    LPS:int
    # In Russian the quantity of syllables in a word is equal to the quantity of vowel letters
    SPS:int

@dataclass
class HighLevelSpeechFeatures:
    loudness: np.float64
    HF_power_ratio:np.float64
    pronounce_speed:PronounceSpeed
    transcription_features:TranscriptionHighLevelFeatures
    # transcription_features:TranscriptionHighLevelFeatures

    @classmethod
    def _HF_power_ratio(
        cls,
        audio:Audio,
        HF_threshold:int = HIGH_FREQUENCY_SPEECH_THRESHOLD,
        )->np.float64:
        # sampling_rate, signal = wavfile.read(file_path)
        # Normalize to [-1, 1]
        
        signal:np.ndarray = audio.data / np.max(np.abs(audio.data))

        # Apply Hann window
        window:np.ndarray = np.hanning(len(signal))
        signal_windowed:np.ndarray = signal * window

        n:int = len(signal_windowed)
        freq_magnitudes:np.ndarray = np.abs(np.fft.fft(signal_windowed))
        freqs:np.ndarray = np.fft.fftfreq(n, d=1/audio.sr)

        # Keep only positive frequencies (half the spectrum)
        positive_freqs:np.ndarray = freqs[:n//2]
        positive_magnitudes:np.ndarray = freq_magnitudes[:n//2]

        # Convert magnitudes to power (energy)
        power_spectrum:np.ndarray = positive_magnitudes ** 2

        total_energy:np.float64 = np.sum(power_spectrum)
        high_freq_mask:np.ndarray = positive_freqs > HF_threshold  # Adjust threshold as needed
        high_freq_energy:np.float64 = np.sum(power_spectrum[high_freq_mask])

        ratio:np.float64 = high_freq_energy / total_energy
        return ratio 

    @classmethod
    def speech_filter(
        cls,
        audio:Audio, 
        low_freq=HUMAN_SPEECH_FREQ_BOTTOM, 
        high_freq=HUMAN_SPEECH_FREQ_TOP,
        )->Audio:

        fft_result:np.ndarray = rfft(audio.data)
        fft_result_filtered:np.ndarray = fft_result.copy()
        freqs:np.ndarray = np.fft.fftfreq(audio.n_frames, d=1.0/audio.sr)

        positive_freqs:np.ndarray = freqs[:len(freqs) // 2 + 1]

        for i, freq in enumerate(positive_freqs):
            if abs(freq) > high_freq or abs(freq) < low_freq:
                fft_result_filtered[i] = 0

        filtered_signal:np.ndarray = irfft(fft_result_filtered)
        sample_dtype:type = audio.sample_dtype()
        filtered_signal:np.ndarray = filtered_signal.astype(sample_dtype) 
        return audio.new_data_copy(data=filtered_signal)
    
    @classmethod
    def _volume(
        cls,
        audio:Audio, 
        data_type:type = np.float64,
        )->np.float64:
        meter:pyln.meter.Meter = pyln.Meter(audio.sr)
        return meter.integrated_loudness(audio.data.astype(data_type))

    @classmethod
    def _pronunciation_speed(
        cls,
        audio:Audio,
        vowels:Set[str] = RUSSIAN_VOWELS,
        transcriber:Callable[
            [
            torch.Tensor, 
            int, 
            WhisperProcessor, 
            WhisperForConditionalGeneration
            ], 
            str
        ] = whisper_tensor_with_sr_transcription,
        ):

        duration:float = librosa.get_duration(y=audio.data, sr=audio.sr)
        transcription:str = audio.joined_transcription(
            transcriber=transcriber,
        )
        word_count:int = len(transcription.split())
        
        return PronounceSpeed(
            WPS=word_count / duration,
            LPS=sum(
                map(
                    lambda l: l.isalpha(), 
                    transcription,
                )
            ) / duration,
            SPS = sum(
                map(
                    lambda letter: letter in vowels,
                    transcription,
                )
            ) / duration,
        )

    @classmethod
    def audio_init(
        cls,
        audio:Audio,
        transcriber:Optional[
            Callable[
                [
                torch.Tensor, 
                int, 
                WhisperProcessor, 
                WhisperForConditionalGeneration
                ], 
                str,
            ]
        ] = whisper_tensor_with_sr_transcription,
        HF_threshold: int = HIGH_FREQUENCY_SPEECH_THRESHOLD,
        vowels:Set[str] = RUSSIAN_VOWELS,
        ): 
        if audio._transcription is None and transcriber is None:
            raise Exception('audio._transcription is None and transcriber is None')
        
        not_nan_quanity:int = np.count_nonzero(~np.isnan(np.array([audio.data])))
        are_all_zeros:bool = not np.any(audio.data)
        if not_nan_quanity == 0 or are_all_zeros:
            return None
        
        return HighLevelSpeechFeatures(
            loudness=cls._volume(
                audio=audio,
            ),
            HF_power_ratio=cls._HF_power_ratio(
                audio=audio,
                HF_threshold=HF_threshold,
            ),
            pronounce_speed=cls._pronunciation_speed(
                audio=audio,
                vowels=vowels, 
                transcriber=transcriber,
            ),
            transcription_features = TranscriptionHighLevelFeatures.text_init(
                text=audio.joined_transcription(
                    transcriber=transcriber,
                ),
            )
        )

    # @classmethod
    # def wav_path_init(
    #     cls,
    #     path:Path,
    #     transcription:Optional[str],
    #     transcriber:Callable[
    #         [
    #         torch.Tensor, 
    #         int, 
    #         WhisperProcessor, 
    #         WhisperForConditionalGeneration
    #         ], 
    #         str,
    #     ] = whisper_tensor_with_sr_transcription,
    #     HF_threshold: int = HIGH_FREQUENCY_SPEECH_THRESHOLD,
    #     vowels:Set[str] = RUSSIAN_VOWELS,
    #     ):
    #     audio:Audio = Audio.wav_file_path_init(
    #         path=path,
    #         transcription=transcription,
    #     )
    #     return cls.audio_init(
    #         audio=audio,
    #         transcriber=transcriber,
    #         HF_threshold=HF_threshold,
    #         vowels=vowels
    #     )
    #     # text:str = transcriber(path)

    #     # return HighLevelSpeechFeatures(
    #     #     loudness=cls._volume(
    #     #         audio=audio,
    #     #     ),
    #     #     HF_power_ratio=cls._HF_power_ratio(
    #     #         audio=audio,
    #     #         HF_threshold=HF_threshold,
    #     #     ),
    #     #     pronounce_speed=cls._pronunciation_speed(
    #     #         audio=audio,
    #     #         vowels=RUSSIAN_VOWELS, 
    #     #         transcriber=transcriber,
    #     #     ),
    #     #     transcription_features = TranscriptionHighLevelFeatures.text_init(
    #     #         text=audio.joined_transcription(
    #     #             transcriber=transcriber,
    #     #         ),
    #     #     )
    #     # )


@dataclass
class HashHLF:
    hash:str
    features:Optional[HighLevelSpeechFeatures]
