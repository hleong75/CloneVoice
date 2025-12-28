"""
CloneVoice - Main Application

Version 1: Voice cloning with manual CSV input
Version 2: Automatic CSV generation from audio files

Usage:
    # Version 1 - With existing CSV
    python clone_voice.py --csv data.csv --audio-dir ./audios --text "Text to speak" --output output.wav
    
    # Version 2 - Automatic CSV generation
    python clone_voice.py --auto --audio-dir ./audios --text "Text to speak" --output output.wav
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


def version1_clone(
    csv_path: str,
    audio_dir: Optional[str],
    text: str,
    output_path: str,
    language: str = "fr",
    use_gpu: bool = True,
    preprocess_audio: bool = True,
    inference_params: Optional[dict] = None
) -> str:
    """
    Version 1: Voice cloning with manual CSV input.
    
    Args:
        csv_path: Path to CSV file with audio IDs and transcriptions.
        audio_dir: Directory containing audio files.
        text: Text to synthesize with cloned voice.
        output_path: Path to save the output audio.
        language: Language code for synthesis.
        use_gpu: Whether to use GPU if available.
        preprocess_audio: Whether to preprocess audio for better quality.
        inference_params: Optional inference parameters for voice synthesis.
        
    Returns:
        Path to the generated audio file.
    """
    from src.csv_parser import CSVParser
    from src.voice_cloner import VoiceCloner
    
    print("=" * 50)
    print("CloneVoice - Version 1 (Manual CSV)")
    print("=" * 50)
    
    # Parse CSV
    print(f"\n📄 Parsing CSV file: {csv_path}")
    parser = CSVParser(audio_directory=audio_dir)
    entries = parser.parse(csv_path)
    
    print(f"   Found {len(entries)} entries")
    
    # Collect reference audio files
    reference_audios = []
    for entry in entries:
        if entry.audio_path:
            reference_audios.append(entry.audio_path)
            print(f"   ✓ Found audio: {entry.audio_id}")
        else:
            print(f"   ✗ Missing audio: {entry.audio_id}")
    
    if not reference_audios:
        raise ValueError("No audio files found! Make sure audio files exist in the specified directory.")
    
    print(f"\n🎤 Using {len(reference_audios)} reference audio(s) for voice cloning")
    
    # Initialize voice cloner with optimized settings
    print("\n🤖 Initializing AI voice cloning model...")
    cloner = VoiceCloner(
        use_gpu=use_gpu,
        preprocess_audio=preprocess_audio,
        inference_params=inference_params
    )
    
    try:
        # Clone voice and generate speech
        print(f"\n🔊 Generating speech with cloned voice...")
        print(f"   Text: \"{text}\"")
        print(f"   Language: {language}")
        
        result = cloner.clone_voice(
            text=text,
            reference_audio=reference_audios,
            output_path=output_path,
            language=language
        )
        
        print(f"\n✅ Audio generated successfully!")
        print(f"   Output: {result}")
        
        return result
    finally:
        cloner.cleanup()


def version2_auto_clone(
    audio_dir: str,
    text: str,
    output_path: str,
    language: str = "fr",
    whisper_model: str = "base",
    use_gpu: bool = True,
    preprocess_audio: bool = True,
    inference_params: Optional[dict] = None
) -> str:
    """
    Version 2: Automatic CSV generation and voice cloning.
    
    This version automatically transcribes audio files using Whisper
    before performing voice cloning.
    
    Args:
        audio_dir: Directory containing audio files.
        text: Text to synthesize with cloned voice.
        output_path: Path to save the output audio.
        language: Language code for synthesis.
        whisper_model: Whisper model size for transcription.
        use_gpu: Whether to use GPU if available.
        preprocess_audio: Whether to preprocess audio for better quality.
        inference_params: Optional inference parameters for voice synthesis.
        
    Returns:
        Path to the generated audio file.
    """
    from src.auto_transcriber import auto_generate_csv
    from src.voice_cloner import VoiceCloner
    
    import shutil
    import tempfile
    
    print("=" * 50)
    print("CloneVoice - Version 2 (Automatic)")
    print("=" * 50)
    
    # Create temporary CSV file
    temp_dir = tempfile.mkdtemp()
    temp_csv = os.path.join(temp_dir, "auto_transcriptions.csv")
    
    cloner = None
    try:
        # Auto-generate CSV with transcriptions
        print(f"\n📝 Automatically transcribing audio files...")
        print(f"   Directory: {audio_dir}")
        print(f"   Whisper model: {whisper_model}")
        
        csv_path = auto_generate_csv(
            audio_directory=audio_dir,
            output_csv=temp_csv,
            model_size=whisper_model,
            language=language
        )
        
        # Get list of audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        audio_files = []
        audio_dir_path = Path(audio_dir)
        
        for ext in audio_extensions:
            audio_files.extend(audio_dir_path.glob(f"*{ext}"))
            audio_files.extend(audio_dir_path.glob(f"*{ext.upper()}"))
        
        reference_audios = [str(f) for f in sorted(audio_files)]
        
        if not reference_audios:
            raise ValueError("No audio files found in directory!")
        
        print(f"\n🎤 Using {len(reference_audios)} reference audio(s) for voice cloning")
        
        # Initialize voice cloner with optimized settings
        print("\n🤖 Initializing AI voice cloning model...")
        cloner = VoiceCloner(
            use_gpu=use_gpu,
            preprocess_audio=preprocess_audio,
            inference_params=inference_params
        )
        
        # Clone voice and generate speech
        print(f"\n🔊 Generating speech with cloned voice...")
        print(f"   Text: \"{text}\"")
        print(f"   Language: {language}")
        
        result = cloner.clone_voice(
            text=text,
            reference_audio=reference_audios,
            output_path=output_path,
            language=language
        )
        
        print(f"\n✅ Audio generated successfully!")
        print(f"   Output: {result}")
        
        return result
    
    finally:
        # Clean up temporary files and cloner resources
        if cloner:
            cloner.cleanup()
        shutil.rmtree(temp_dir, ignore_errors=True)


def batch_clone(
    csv_path: str,
    audio_dir: Optional[str],
    texts: List[str],
    output_dir: str,
    language: str = "fr",
    use_gpu: bool = True
) -> List[str]:
    """
    Batch processing: Generate multiple audio files with cloned voice.
    
    Args:
        csv_path: Path to CSV file with audio IDs and transcriptions.
        audio_dir: Directory containing audio files.
        texts: List of texts to synthesize.
        output_dir: Directory to save output audio files.
        language: Language code for synthesis.
        use_gpu: Whether to use GPU if available.
        
    Returns:
        List of paths to generated audio files.
    """
    from src.csv_parser import CSVParser
    from src.voice_cloner import VoiceCloner
    
    print("=" * 50)
    print("CloneVoice - Batch Processing")
    print("=" * 50)
    
    # Parse CSV
    print(f"\n📄 Parsing CSV file: {csv_path}")
    parser = CSVParser(audio_directory=audio_dir)
    entries = parser.parse(csv_path)
    
    # Collect reference audio files
    reference_audios = [e.audio_path for e in entries if e.audio_path]
    
    if not reference_audios:
        raise ValueError("No audio files found!")
    
    print(f"   Using {len(reference_audios)} reference audio(s)")
    
    # Initialize voice cloner
    print("\n🤖 Initializing AI voice cloning model...")
    cloner = VoiceCloner(use_gpu=use_gpu)
    
    # Generate audio files
    print(f"\n🔊 Generating {len(texts)} audio files...")
    
    results = cloner.batch_clone(
        texts=texts,
        reference_audio=reference_audios,
        output_dir=output_dir,
        language=language
    )
    
    print(f"\n✅ Batch processing complete!")
    print(f"   Generated {len(results)} audio files")
    
    return results


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="CloneVoice - AI-Powered Voice Cloning Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Version 1: Clone voice using manual CSV
  python clone_voice.py --csv data.csv --audio-dir ./audios --text "Bonjour le monde" --output output.wav

  # Version 2: Automatic transcription and cloning
  python clone_voice.py --auto --audio-dir ./audios --text "Bonjour le monde" --output output.wav

  # Batch processing with text file
  python clone_voice.py --csv data.csv --audio-dir ./audios --text-file texts.txt --output-dir ./outputs
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Version 2: Automatically transcribe audio files (no CSV needed)"
    )
    
    # Input options
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file with audio IDs and transcriptions (Version 1)"
    )
    
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory containing reference audio files"
    )
    
    # Text options
    parser.add_argument(
        "--text",
        type=str,
        help="Text to synthesize with cloned voice"
    )
    
    parser.add_argument(
        "--text-file",
        type=str,
        help="File containing texts to synthesize (one per line, for batch mode)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output audio file path (single file mode)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch mode"
    )
    
    # Model options
    parser.add_argument(
        "--language",
        type=str,
        default="fr",
        help="Language code for synthesis (default: fr)"
    )
    
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for auto-transcription (default: base)"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    # Quality options
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable audio preprocessing (not recommended)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.65,
        help="Inference temperature (0.1-1.0, default: 0.65). Lower = more deterministic"
    )
    
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (0.5-2.0, default: 1.0)"
    )
    
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=5.0,
        help="Repetition penalty (1.0-10.0, default: 5.0). Higher = less repetition"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.auto and not args.csv:
        parser.error("Either --csv or --auto must be specified")
    
    if not args.text and not args.text_file:
        parser.error("Either --text or --text-file must be specified")
    
    if args.text_file and not args.output_dir:
        parser.error("--output-dir is required when using --text-file")
    
    if args.text and not args.output:
        parser.error("--output is required when using --text")
    
    use_gpu = not args.no_gpu
    preprocess_audio = not args.no_preprocess
    
    # Build inference parameters
    inference_params = {
        "temperature": args.temperature,
        "speed": args.speed,
        "repetition_penalty": args.repetition_penalty
    }
    
    try:
        if args.text_file:
            # Batch mode
            with open(args.text_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            if args.auto:
                # Auto mode doesn't support batch directly, run single for each
                from src.auto_transcriber import auto_generate_csv
                import tempfile
                
                temp_dir = tempfile.mkdtemp()
                temp_csv = os.path.join(temp_dir, "auto.csv")
                auto_generate_csv(args.audio_dir, temp_csv, args.whisper_model, args.language)
                
                batch_clone(
                    csv_path=temp_csv,
                    audio_dir=args.audio_dir,
                    texts=texts,
                    output_dir=args.output_dir,
                    language=args.language,
                    use_gpu=use_gpu
                )
            else:
                batch_clone(
                    csv_path=args.csv,
                    audio_dir=args.audio_dir,
                    texts=texts,
                    output_dir=args.output_dir,
                    language=args.language,
                    use_gpu=use_gpu
                )
        else:
            # Single file mode
            if args.auto:
                version2_auto_clone(
                    audio_dir=args.audio_dir,
                    text=args.text,
                    output_path=args.output,
                    language=args.language,
                    whisper_model=args.whisper_model,
                    use_gpu=use_gpu,
                    preprocess_audio=preprocess_audio,
                    inference_params=inference_params
                )
            else:
                version1_clone(
                    csv_path=args.csv,
                    audio_dir=args.audio_dir,
                    text=args.text,
                    output_path=args.output,
                    language=args.language,
                    use_gpu=use_gpu,
                    preprocess_audio=preprocess_audio,
                    inference_params=inference_params
                )
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
