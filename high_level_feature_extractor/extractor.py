from dataclasses import (
    dataclass,
    asdict,
    is_dataclass,
    fields,
)
import numpy as np
import pandas as pd
from pathlib import Path
import pyloudnorm as pyln
from typing import (
    Callable,
    Optional, 
    Set,
    List,
    Dict,
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

from high_level_feature_extractor.volume.human_speech import (
    HIGH_FREQUENCY_SPEECH_THRESHOLD,
)
from high_level_feature_extractor.volume.human_speech import (
    HUMAN_SPEECH_FREQ_BOTTOM,
    HUMAN_SPEECH_FREQ_TOP,
    HIGH_FREQUENCY_SPEECH_THRESHOLD,
)

from high_level_feature_extractor.text.all import (
    TranscriptionHighLevelFeatures,
)

from configs.base import (
    RUSSIAN_VOWELS,
    EPSILON,
)

from utils.dataclass import (
    flatten_dict,
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
        if audio.sr == 0:
            return np.nan
        freqs:np.ndarray = np.fft.fftfreq(n, d=1/(audio.sr))

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
        if audio.sr == 0:
            return np.nan
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
        ):

        duration:float = librosa.get_duration(y=audio.data, sr=audio.sr)
        transcription:str = audio.joined_transcription()
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
        filter_speech:bool=True,
        HF_threshold: int = HIGH_FREQUENCY_SPEECH_THRESHOLD,
        vowels:Set[str] = RUSSIAN_VOWELS,
        ): 
        if filter_speech:
            audio = cls.speech_filter(audio=audio)
        if audio._transcription is None:
            raise Exception('audio._transcription is None')

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
            ),
            transcription_features = TranscriptionHighLevelFeatures.text_init(
                text=audio.joined_transcription(),
            )
        )



@dataclass
class HashHLF:
    hash:str
    features:Optional[HighLevelSpeechFeatures]


def hash_HLF_list_2_df(
    l:List[HashHLF],
    )->pd.DataFrame:
    l = list(filter(lambda x: x.features is not None, l))
    
    df:pd.DataFrame = pd.DataFrame(
        index=map(
            lambda x: x.hash, 
            l,
        ), 
        data=filter(
            lambda x: x is not None,
            map(
                lambda x: flatten_dict(
                        flatten_dict(
                        asdict(
                            x.features,
                        )
                    ) 
                ) if x.features is not None else None, 
                l,
            ),
        )
    )
    return df

HLF_ENG_NAME_2_RU_NAME:Dict[str,str] = {
    'loudness': 'средняя громкость',
    'HF_power_ratio': 'доля мощности высокочастотных составляющих речи',
    'pronounce_speed_WPS': 'средняя скорость произношения (слов в секунду)',
    'pronounce_speed_LPS': 'средняя скорость произношения (букв в секунду)',
    'pronounce_speed_SPS': 'средняя скорость произношения (слогов в секунду)',
    'transcription_features_mean_words_length': 'средняя длина слова',
    'transcription_features_profanity_words_ratio': 'доля ненормативных слов',
    'transcription_features_meaning': 'средняя значимость слова',
    'transcription_features_POS_ratio_ADVB': 'доля наречий (кроме наречий времени)',
    'transcription_features_POS_ratio_COMP': 'доля компаративов', # Слова "лучше", "получше", "выше" относятся к сравнительной степени прилагательных и наречий. Сравнительная степень прилагательного "хороший" (пример: Этот вариант лучше → прилагательное). Сравнительная степень наречия "хорошо" (пример: Она поёт лучше → наречие).
    'transcription_features_POS_ratio_CONJ': 'доля союзов',
    'transcription_features_POS_ratio_GRND': 'доля деепричастий',
    'transcription_features_POS_ratio_INFN': 'доля глаголов-инфинитивов',
    'transcription_features_POS_ratio_INTJ': 'доля междометий',
    'transcription_features_POS_ratio_PRCL': 'доля частиц',
    'transcription_features_POS_ratio_PRED': 'доля предикативов', # TODO: ensure it is Union["Наречие времени", "Наречие времени (устаревшее)"]
    'transcription_features_POS_ratio_PREP': 'доля предлогов',
    'transcription_features_POS_ratio_VERB': 'доля глаголов в личной форме',
    'transcription_features_POS_ratio_ADJS': 'доля кратких прилагательных',
    'transcription_features_POS_ratio_PRTS': 'доля кратких причастий',
    'transcription_features_POS_ratio_NPRO': 'доля местоимений-существительных',
    'transcription_features_POS_ratio_NOUN': 'доля существительных',
    'transcription_features_POS_ratio_ADJF': 'доля полных прилагательных',
    'transcription_features_POS_ratio_NUMR': 'доля числительных',
    'transcription_features_POS_ratio_PRTF': 'доля полных причастий',
    'transcription_features_POS_ratio_NONE': 'доля слов с неопределенной частью речи',
}

HLF_ENG_2_RU_RENAMER:Callable[[str], str] = lambda x: HLF_ENG_NAME_2_RU_NAME[x]
