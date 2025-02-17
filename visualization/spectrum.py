import librosa
import numpy as np
import matplotlib.pyplot as plt

def audio_path_2_show_spectrum(
        audio_path, 
        frame_index=0,
        )->None:
    """
    Plots the magnitude spectrum of an audio file for a given frame.

    Parameters:
    - audio_path (str): Path to the audio file.
    - frame_index (int): Index of the frame for which to plot the spectrum.
                         Defaults to 0 (the first frame).
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Compute the Short-Time Fourier Transform (STFT)
    stft:np.ndarray = librosa.stft(y)


    # Calculate magnitude spectrum
    magnitude_spectrum:np.ndarray = np.abs(stft)

    # Get the magnitude spectrum for the specified frame
    frame:np.ndarray = magnitude_spectrum[:, frame_index]

    # Generate frequency axis
    freqs:np.ndarray = librosa.fft_frequencies(sr=sr)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, frame)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда (дБ)')
    plt.title(f'Спектр')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


