# voice_analyzer/audio.py
from scipy.ndimage.morphology import binary_dilation
from voice_analyzer.hparams import *
from pathlib import Path
from typing import Optional, Union
import numpy as np
import librosa

int16_max = (2 ** 15) - 1

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int] = None):
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav

    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate)

    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_silences_by_energy(wav)
    return wav

def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    """
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T

def trim_silences_by_energy(wav, energy_threshold=0.05, window_size_ms=100, padding_ms=50):
    """
    A simple energy-based silence trimmer.
    """
    window_size = int(sampling_rate * window_size_ms / 1000)
    padding_samples = int(sampling_rate * padding_ms / 1000)
    
    wav_padded = np.pad(wav, padding_samples, mode='constant')
    
    energies = []
    for i in range(0, len(wav_padded) - window_size, window_size):
        window = wav_padded[i:i + window_size]
        energies.append(np.sqrt(np.mean(window**2)))
    
    is_speech = np.array(energies) > energy_threshold
    
    mask = np.zeros(len(wav_padded), dtype=bool)
    for i, speech in enumerate(is_speech):
        if speech:
            start_index = i * window_size
            end_index = start_index + window_size
            mask[start_index:end_index] = True
            
    mask = binary_dilation(mask, np.ones(padding_samples * 2))
    
    return wav_padded[mask]

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase_only and decrease_only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max) if rms > 0 else -np.inf
    dBFS_change = target_dBFS - wave_dBFS
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))