"""
Automatic Transcription Module

Uses OpenAI Whisper for automatic speech recognition.
This module enables automatic generation of transcriptions from audio files.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class AutoTranscriber:
    """
    Automatic speech recognition using OpenAI Whisper.
    
    Whisper is a state-of-the-art ASR model that supports:
    - 99+ languages
    - Robust transcription even with background noise
    - Automatic language detection
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None
    ):
        """
        Initialize the transcriber.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                       Larger models are more accurate but slower.
            device: Specific device to use ('cuda', 'cpu').
            language: Language code for transcription. If None, auto-detects.
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None
        self._initialized = False
    
    def _initialize(self) -> None:
        """Lazy initialization of the Whisper model."""
        if self._initialized:
            return
        
        print(f"Loading Whisper {self.model_size} model...")
        
        import whisper
        
        # Determine device
        if self.device:
            device = self.device
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using GPU for transcription")
        else:
            device = "cpu"
            print("Using CPU for transcription")
        
        self.model = whisper.load_model(self.model_size, device=device)
        self._initialized = True
        
        print("Whisper model loaded successfully!")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file.
            language: Override language for this transcription.
            
        Returns:
            Dictionary with transcription results including:
            - text: Full transcription
            - segments: List of segments with timestamps
            - language: Detected/used language
        """
        self._initialize()
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Use instance language or provided language
        lang = language or self.language
        
        result = self.model.transcribe(
            str(audio_path),
            language=lang,
            fp16=torch.cuda.is_available()
        )
        
        return result
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files.
            language: Language code for transcription.
            
        Returns:
            List of transcription result dictionaries.
        """
        from tqdm import tqdm
        
        results = []
        
        for audio_path in tqdm(audio_paths, desc="Transcribing"):
            try:
                result = self.transcribe(audio_path, language)
                results.append({
                    'audio_path': audio_path,
                    'text': result['text'].strip(),
                    'language': result.get('language', 'unknown'),
                    'success': True
                })
            except Exception as e:
                results.append({
                    'audio_path': audio_path,
                    'text': '',
                    'error': str(e),
                    'success': False
                })
        
        return results


def auto_generate_csv(
    audio_directory: str,
    output_csv: str,
    model_size: str = "base",
    language: Optional[str] = None
) -> str:
    """
    Automatically generate a CSV file with transcriptions from audio files.
    
    This function scans a directory for audio files, transcribes them using
    Whisper, and creates a CSV file with audio IDs and transcriptions.
    
    Args:
        audio_directory: Directory containing audio files.
        output_csv: Path to save the generated CSV file.
        model_size: Whisper model size.
        language: Language code for transcription.
        
    Returns:
        Path to the generated CSV file.
    """
    import csv
    from pathlib import Path
    from tqdm import tqdm
    
    audio_dir = Path(audio_directory)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Directory not found: {audio_directory}")
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_directory}")
    
    # Sort files for consistent ordering
    audio_files = sorted(audio_files)
    
    print(f"Found {len(audio_files)} audio files")
    
    # Initialize transcriber
    transcriber = AutoTranscriber(model_size=model_size, language=language)
    
    # Transcribe all files
    entries = []
    
    for audio_file in tqdm(audio_files, desc="Transcribing audio files"):
        try:
            result = transcriber.transcribe(str(audio_file))
            text = result['text'].strip()
            
            if text:
                entries.append({
                    'audio_id': audio_file.stem,
                    'transcription': text,
                    'audio_path': str(audio_file)
                })
                
        except Exception as e:
            print(f"Error transcribing {audio_file.name}: {e}")
            continue
    
    if not entries:
        raise ValueError("No transcriptions could be generated")
    
    # Write CSV file
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['audio_id', 'transcription'])
        
        for entry in entries:
            writer.writerow([entry['audio_id'], entry['transcription']])
    
    print(f"CSV file saved to: {output_csv}")
    print(f"Total entries: {len(entries)}")
    
    return str(output_path.absolute())


def create_auto_transcriber(
    model_size: str = "base",
    language: Optional[str] = None
) -> AutoTranscriber:
    """
    Factory function to create an AutoTranscriber instance.
    
    Args:
        model_size: Whisper model size.
        language: Language code for transcription.
        
    Returns:
        Configured AutoTranscriber instance.
    """
    return AutoTranscriber(model_size=model_size, language=language)
