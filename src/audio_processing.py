"""
Audio Processing Module

Handles audio loading, preprocessing, and saving.
Includes advanced audio preprocessing for optimal voice cloning quality.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


# Constants for optimal voice cloning
OPTIMAL_SAMPLE_RATE = 22050
MIN_DURATION_SECONDS = 3.0
MAX_DURATION_SECONDS = 30.0
OPTIMAL_DURATION_SECONDS = 10.0


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


def preprocess_audio_for_cloning(
    audio_path: str,
    output_path: Optional[str] = None,
    target_sr: int = OPTIMAL_SAMPLE_RATE,
    normalize: bool = True,
    trim_silence: bool = True,
    reduce_noise: bool = True,
    max_duration: Optional[float] = None
) -> str:
    """
    Preprocess audio file for optimal voice cloning quality.
    
    This function applies several preprocessing steps to improve the quality
    of voice cloning:
    - Resampling to optimal sample rate (22050 Hz)
    - Normalization to consistent volume
    - Silence trimming to remove dead air
    - Optional noise reduction
    
    Args:
        audio_path: Path to the input audio file.
        output_path: Path to save preprocessed audio. If None, creates temp file.
        target_sr: Target sample rate for output.
        normalize: If True, normalize audio volume.
        trim_silence: If True, trim leading/trailing silence.
        reduce_noise: If True, apply light noise reduction.
        max_duration: Maximum duration in seconds. If audio is longer, it will be trimmed.
        
    Returns:
        Path to the preprocessed audio file.
    """
    import librosa
    import soundfile as sf
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    
    # Trim silence from beginning and end
    if trim_silence:
        audio, _ = librosa.effects.trim(
            audio,
            top_db=30,  # Threshold in decibels below reference
            frame_length=2048,
            hop_length=512
        )
    
    # Apply max duration trimming
    if max_duration is not None:
        max_samples = int(max_duration * target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
    
    # Light noise reduction using spectral gating
    if reduce_noise:
        audio = _reduce_noise_spectral(audio, sr)
    
    # Normalize audio
    if normalize:
        audio = _normalize_audio(audio)
    
    # Determine output path
    if output_path is None:
        # Create temporary file
        fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save preprocessed audio
    sf.write(str(output_path), audio, target_sr)
    
    return str(output_path)


def _normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to a target dB level.
    
    Args:
        audio: Audio data as numpy array.
        target_db: Target peak level in decibels (negative value).
        
    Returns:
        Normalized audio data.
    """
    # Compute current peak
    peak = np.max(np.abs(audio))
    
    if peak < 1e-10:
        # Audio is silent or near-silent
        return audio
    
    # Convert target dB to linear scale
    target_linear = 10 ** (target_db / 20)
    
    # Normalize
    normalized = audio * (target_linear / peak)
    
    # Ensure no clipping
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def _reduce_noise_spectral(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply light spectral noise reduction.
    
    Uses spectral gating to reduce background noise while preserving speech.
    
    Args:
        audio: Audio data as numpy array.
        sr: Sample rate.
        
    Returns:
        Noise-reduced audio data.
    """
    import librosa
    
    # Compute short-time Fourier transform
    n_fft = 2048
    hop_length = 512
    
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise floor from quietest frames
    # Use bottom 10% of frames as noise estimate
    frame_energy = np.sum(magnitude ** 2, axis=0)
    noise_frames = np.argsort(frame_energy)[:max(1, len(frame_energy) // 10)]
    noise_profile = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
    
    # Apply spectral gating
    # Subtract noise profile (with floor to prevent negative values)
    gate_threshold = 1.5  # How aggressive the gating is
    mask = magnitude / (noise_profile * gate_threshold + 1e-10)
    mask = np.minimum(mask, 1.0)  # Cap at 1.0
    
    # Apply soft mask
    magnitude_cleaned = magnitude * mask
    
    # Reconstruct audio
    stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
    audio_cleaned = librosa.istft(stft_cleaned, hop_length=hop_length, length=len(audio))
    
    return audio_cleaned


def preprocess_reference_audios(
    audio_paths: list,
    output_dir: Optional[str] = None,
    **preprocess_kwargs
) -> list:
    """
    Preprocess multiple reference audio files for voice cloning.
    
    Args:
        audio_paths: List of paths to audio files.
        output_dir: Directory to save preprocessed files. If None, uses temp directory.
        **preprocess_kwargs: Additional arguments passed to preprocess_audio_for_cloning.
        
    Returns:
        List of paths to preprocessed audio files.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='clonevoice_preprocessed_')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessed_paths = []
    
    for audio_path in audio_paths:
        audio_path = Path(audio_path)
        output_path = output_dir / f"preprocessed_{audio_path.stem}.wav"
        
        try:
            result = preprocess_audio_for_cloning(
                str(audio_path),
                str(output_path),
                **preprocess_kwargs
            )
            preprocessed_paths.append(result)
        except Exception as e:
            print(f"Warning: Failed to preprocess {audio_path.name}: {e}")
            # Use original file as fallback
            preprocessed_paths.append(str(audio_path))
    
    return preprocessed_paths


def validate_and_get_best_reference(
    audio_paths: list,
    min_duration: float = MIN_DURATION_SECONDS,
    max_duration: float = MAX_DURATION_SECONDS
) -> Tuple[list, list]:
    """
    Validate audio files and return the best references for voice cloning.
    
    Filters out audio files that are too short or have issues,
    and returns a sorted list with the best candidates first.
    
    Args:
        audio_paths: List of paths to audio files.
        min_duration: Minimum acceptable duration.
        max_duration: Maximum acceptable duration.
        
    Returns:
        Tuple of (valid_paths, messages) where valid_paths are sorted by quality
        and messages contain validation info.
    """
    valid_paths = []
    messages = []
    
    for audio_path in audio_paths:
        try:
            duration = get_audio_duration(audio_path)
            
            if duration < min_duration:
                messages.append(f"❌ {Path(audio_path).name}: Too short ({duration:.1f}s < {min_duration}s)")
                continue
            
            if duration > max_duration:
                # Still valid but will be trimmed
                messages.append(f"⚠️ {Path(audio_path).name}: Long ({duration:.1f}s), will be trimmed to {max_duration}s")
            else:
                messages.append(f"✓ {Path(audio_path).name}: Good quality ({duration:.1f}s)")
            
            valid_paths.append((audio_path, duration))
            
        except Exception as e:
            messages.append(f"❌ {Path(audio_path).name}: Error - {e}")
    
    # Sort by duration (prefer files closer to optimal duration of 10s)
    valid_paths.sort(key=lambda x: abs(x[1] - OPTIMAL_DURATION_SECONDS))
    
    return [p[0] for p in valid_paths], messages
