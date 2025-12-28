"""
CloneVoice - Simple Python API

Provides a clean, simple interface for voice cloning operations.

Example usage:
    from src.api import clone_voice, clone_voice_auto
    
    # Version 1: With CSV
    clone_voice(
        csv_path="data.csv",
        audio_dir="./audios",
        text="Hello world",
        output_path="output.wav"
    )
    
    # Version 2: Automatic
    clone_voice_auto(
        audio_dir="./audios",
        text="Hello world",
        output_path="output.wav"
    )
"""

from pathlib import Path
from typing import List, Optional, Union


def clone_voice(
    csv_path: str,
    audio_dir: str,
    text: str,
    output_path: str,
    language: str = "fr",
    use_gpu: bool = True
) -> str:
    """
    Clone a voice using a CSV file with audio references and transcriptions.
    
    This is Version 1 of CloneVoice - requires a manually created CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: audio_id, transcription
        audio_dir: Directory containing the reference audio files
        text: Text to synthesize with the cloned voice
        output_path: Where to save the generated audio
        language: Language code (default: "fr" for French)
        use_gpu: Use GPU acceleration if available
        
    Returns:
        Path to the generated audio file
        
    Example:
        >>> result = clone_voice(
        ...     csv_path="voices.csv",
        ...     audio_dir="./my_voice_samples",
        ...     text="Bonjour, comment allez-vous?",
        ...     output_path="greeting.wav"
        ... )
        >>> print(f"Generated: {result}")
    """
    from src.csv_parser import CSVParser
    from src.voice_cloner import VoiceCloner
    
    # Parse CSV to get audio entries
    parser = CSVParser(audio_directory=audio_dir)
    entries = parser.parse(csv_path)
    
    # Collect reference audio paths
    reference_audios = [e.audio_path for e in entries if e.audio_path]
    
    if not reference_audios:
        raise ValueError(
            "No audio files found. Ensure audio files exist in the specified directory "
            "and their names match the audio_id column in the CSV."
        )
    
    # Clone voice
    cloner = VoiceCloner(use_gpu=use_gpu)
    return cloner.clone_voice(
        text=text,
        reference_audio=reference_audios,
        output_path=output_path,
        language=language
    )


def clone_voice_auto(
    audio_dir: str,
    text: str,
    output_path: str,
    language: str = "fr",
    whisper_model: str = "base",
    use_gpu: bool = True
) -> str:
    """
    Clone a voice with automatic transcription - no CSV needed.
    
    This is Version 2 of CloneVoice - automatically transcribes audio files
    using OpenAI Whisper before voice cloning.
    
    Args:
        audio_dir: Directory containing the reference audio files
        text: Text to synthesize with the cloned voice
        output_path: Where to save the generated audio
        language: Language code (default: "fr" for French)
        whisper_model: Whisper model size ("tiny", "base", "small", "medium", "large")
        use_gpu: Use GPU acceleration if available
        
    Returns:
        Path to the generated audio file
        
    Example:
        >>> result = clone_voice_auto(
        ...     audio_dir="./my_voice_samples",
        ...     text="Bonjour, comment allez-vous?",
        ...     output_path="greeting.wav"
        ... )
        >>> print(f"Generated: {result}")
    """
    from src.voice_cloner import VoiceCloner
    
    audio_dir_path = Path(audio_dir)
    
    if not audio_dir_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(audio_dir_path.glob(f"*{ext}"))
        audio_files.extend(audio_dir_path.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")
    
    reference_audios = [str(f) for f in sorted(audio_files)]
    
    # Clone voice
    cloner = VoiceCloner(use_gpu=use_gpu)
    return cloner.clone_voice(
        text=text,
        reference_audio=reference_audios,
        output_path=output_path,
        language=language
    )


def generate_transcriptions(
    audio_dir: str,
    output_csv: str,
    language: Optional[str] = None,
    whisper_model: str = "base"
) -> str:
    """
    Generate a CSV file with automatic transcriptions of audio files.
    
    Useful for creating the CSV file needed by Version 1, or for
    reviewing/editing auto-generated transcriptions before voice cloning.
    
    Args:
        audio_dir: Directory containing audio files to transcribe
        output_csv: Path to save the generated CSV file
        language: Language code for transcription (None = auto-detect)
        whisper_model: Whisper model size
        
    Returns:
        Path to the generated CSV file
        
    Example:
        >>> csv_path = generate_transcriptions(
        ...     audio_dir="./voice_samples",
        ...     output_csv="transcriptions.csv",
        ...     language="fr"
        ... )
        >>> print(f"CSV saved to: {csv_path}")
    """
    from src.auto_transcriber import auto_generate_csv
    
    return auto_generate_csv(
        audio_directory=audio_dir,
        output_csv=output_csv,
        model_size=whisper_model,
        language=language
    )


def batch_clone(
    csv_path: str,
    audio_dir: str,
    texts: List[str],
    output_dir: str,
    language: str = "fr",
    use_gpu: bool = True
) -> List[str]:
    """
    Generate multiple audio files with the cloned voice.
    
    Args:
        csv_path: Path to CSV file with audio references
        audio_dir: Directory containing reference audio files
        texts: List of texts to synthesize
        output_dir: Directory to save generated audio files
        language: Language code for synthesis
        use_gpu: Use GPU acceleration if available
        
    Returns:
        List of paths to generated audio files
        
    Example:
        >>> texts = ["First sentence.", "Second sentence.", "Third sentence."]
        >>> results = batch_clone(
        ...     csv_path="voices.csv",
        ...     audio_dir="./samples",
        ...     texts=texts,
        ...     output_dir="./outputs"
        ... )
    """
    from src.csv_parser import CSVParser
    from src.voice_cloner import VoiceCloner
    
    # Parse CSV
    parser = CSVParser(audio_directory=audio_dir)
    entries = parser.parse(csv_path)
    
    # Collect reference audios
    reference_audios = [e.audio_path for e in entries if e.audio_path]
    
    if not reference_audios:
        raise ValueError("No audio files found")
    
    # Clone voice
    cloner = VoiceCloner(use_gpu=use_gpu)
    return cloner.batch_clone(
        texts=texts,
        reference_audio=reference_audios,
        output_dir=output_dir,
        language=language
    )


# Convenience aliases
v1_clone = clone_voice
v2_clone = clone_voice_auto
