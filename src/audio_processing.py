"""
Audio Processing Module

Handles audio loading, preprocessing, and saving.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def load_audio(
    audio_path: str,
    target_sr: int = 22050,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and optionally resample it.
    
    Args:
        audio_path: Path to the audio file.
        target_sr: Target sample rate. If None, uses original sample rate.
        mono: If True, convert to mono.
        
    Returns:
        Tuple of (audio_data, sample_rate).
        
    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        ValueError: If the audio file format is not supported.
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        import librosa
        
        audio, sr = librosa.load(
            str(audio_path),
            sr=target_sr,
            mono=mono
        )
        
        return audio, sr
        
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")


def save_audio(
    audio_data: np.ndarray,
    output_path: str,
    sample_rate: int = 22050
) -> str:
    """
    Save audio data to a file.
    
    Args:
        audio_data: Audio data as numpy array.
        output_path: Path to save the audio file.
        sample_rate: Sample rate of the audio.
        
    Returns:
        Absolute path to the saved file.
    """
    import soundfile as sf
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize audio to prevent clipping
    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    sf.write(str(output_path), audio_data, sample_rate)
    
    return str(output_path.absolute())


def concatenate_audio(
    audio_files: list,
    output_path: str,
    target_sr: int = 22050,
    silence_duration: float = 0.5
) -> str:
    """
    Concatenate multiple audio files with optional silence between them.
    
    Args:
        audio_files: List of paths to audio files.
        output_path: Path to save the concatenated audio.
        target_sr: Target sample rate.
        silence_duration: Duration of silence between clips in seconds.
        
    Returns:
        Path to the concatenated audio file.
    """
    if not audio_files:
        raise ValueError("No audio files provided")
    
    silence_samples = int(target_sr * silence_duration)
    silence = np.zeros(silence_samples)
    
    audio_parts = []
    
    for audio_file in audio_files:
        audio, _ = load_audio(audio_file, target_sr=target_sr)
        audio_parts.append(audio)
        audio_parts.append(silence)
    
    # Remove the last silence
    if audio_parts:
        audio_parts = audio_parts[:-1]
    
    combined = np.concatenate(audio_parts)
    
    return save_audio(combined, output_path, target_sr)


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_path: Path to the audio file.
        
    Returns:
        Duration in seconds.
    """
    import librosa
    
    return librosa.get_duration(path=audio_path)


def validate_audio_for_cloning(audio_path: str, min_duration: float = 3.0) -> Tuple[bool, str]:
    """
    Validate if an audio file is suitable for voice cloning.
    
    Args:
        audio_path: Path to the audio file.
        min_duration: Minimum required duration in seconds.
        
    Returns:
        Tuple of (is_valid, message).
    """
    try:
        duration = get_audio_duration(audio_path)
        
        if duration < min_duration:
            return False, f"Audio too short ({duration:.1f}s). Minimum {min_duration}s required."
        
        if duration > 30:
            return True, f"Audio is long ({duration:.1f}s). Consider using shorter clips for better results."
        
        return True, f"Audio duration OK ({duration:.1f}s)"
        
    except Exception as e:
        return False, f"Error validating audio: {e}"
