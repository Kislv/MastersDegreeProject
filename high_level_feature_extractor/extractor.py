from dataclasses import (
    dataclass,
)
import numpy as np
from pathlib import Path
from typing import (
    Callable,
    Optional, 
)
import sys
from scipy.io import wavfile
from audio import Audio
from scipy.fft import rfft, irfft
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

# from high_level_feature_extractor.text.all import (
#     TranscriptionHighLevelFeatures,
# )


@dataclass
class HighLevelSpeechFeatures:
    loudness: np.float64
    HF_power_ratio:np.float64
    # transcription_features:TranscriptionHighLevelFeatures

    @classmethod
    def wav_path_2_HF_power_ratio(
        cls,
        file_path:Path,
        HF_threshold:int = HIGH_FREQUENCY_SPEECH_THRESHOLD,
        )->np.float64:
        sampling_rate, signal = wavfile.read(file_path)
        # Normalize to [-1, 1]
        signal:np.ndarray = signal / np.max(np.abs(signal))

        # Apply Hann window
        window:np.ndarray = np.hanning(len(signal))
        signal_windowed:np.ndarray = signal * window

        n:int = len(signal_windowed)
        freq_magnitudes:np.ndarray = np.abs(np.fft.fft(signal_windowed))
        freqs:np.ndarray = np.fft.fftfreq(n, d=1/sampling_rate)

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
    def wav_path_init(
        cls,
        path:Path,
        transcriber:Callable[[Path], str] = whisper_audio_file_2_transcription,
        transcription:Optional[str] = None,
        ):
        audio:Audio = Audio.wav_file_path_init(
            path=path,
        )
        # text:str = transcriber(path)

        return HighLevelSpeechFeatures(
            loudness=audio.volume(),
            HF_power_ratio=cls.wav_path_2_HF_power_ratio(file_path=path)
        )


