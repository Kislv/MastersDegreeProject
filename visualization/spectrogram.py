import torch
import torchaudio
from typing import (
    Union,
)
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

MEL_IN_LOG_DIVIDER:int = 700
MEL_MULTIPLIER:int = 2595
DECIMAL_BASE:int = 10


def compute_spectrogram(
        waveform, 
        hann_window_size:int = 1024,
        n_fft:int = 1024,
        hop_length:int = 512,
        epsilon:float = 1e-10,
        decimal_base:int = DECIMAL_BASE
        ):
    # Create a Hanning window
    window:torch.Tensor = torch.hann_window(hann_window_size)
    
    # Compute STFT
    spectrogram = torch.stft(
        waveform.squeeze(0), 
        n_fft=n_fft, 
        hop_length=hop_length,
        window=window, 
        return_complex=True
    )
    
    # Get magnitude
    magnitude = torch.abs(spectrogram)
    
    # Convert to decibels
    magnitude_db = decimal_base * torch.log10(magnitude + epsilon)  # Adding a small value to avoid log(0)
    
    return magnitude_db

def sr_2_mel_max(
    sr: Union[int, float],
    multiplier:int = MEL_MULTIPLIER,
    in_log_divider:int = MEL_IN_LOG_DIVIDER,
    )->float:
    return multiplier * np.log10(1 + (sr / in_log_divider))

def mel_scale(
    sample_rate:int, 
    n_mels:int=128,
    decimal_base:int=DECIMAL_BASE,
    )->np.ndarray:
    """Convert frequencies to Mel scale."""
    mel_min:int = 0
    mel_max:int = sr_2_mel_max(sample_rate)
    
    mel_bins:np.ndarray = np.linspace(mel_min, mel_max, n_mels + 2)
    
    # Correcting hz_bins calculation to ensure it returns a consistent shape
    hz_bins:np.ndarray = MEL_IN_LOG_DIVIDER * (decimal_base**(mel_bins / MEL_MULTIPLIER) - 1)
    
    return hz_bins

def plot_spectrogram(
        spectrogram:torch.Tensor, 
        sample_rate:int,
        ):
    # Convert spectrogram to numpy array for plotting
    spectrogram_np:np.ndarray = spectrogram.numpy()
    
    # Define Mel scale frequency bins using the correct sample_rate directly
    mel_frequencies:np.ndarray = mel_scale(sample_rate)

    # Time axis in seconds
    time_axis = np.arange(spectrogram_np.shape[1]) * (512 / sample_rate)

    plt.figure(figsize=(12, 6))
    
    # Adjusting the extent parameters for correct display
    plt.imshow(spectrogram_np, aspect='auto', origin='lower', cmap='inferno', extent=[time_axis.min(), time_axis.max(), mel_frequencies[0], mel_frequencies[-1]])
    
    plt.title('Спектрограмма')
    plt.xlabel('Время (секунды)')
    plt.ylabel('Частота (Мел-шкала)')
    plt.colorbar(label='Магнитуда (дБ)')
    plt.show()

def audio_path_2_show_spectrogram(
    audio_path:Path,
    ):
    waveform, sample_rate = torchaudio.load(audio_path)
    spectrogram = compute_spectrogram(waveform=waveform)
    plot_spectrogram(spectrogram, sample_rate)
