#!/usr/bin/env python3
"""
Audio Quality Test Script for CloneVoice

This script tests the voice cloning program with the audio files in the audios directory
and saves the generated audio files in the outputs directory.

Usage:
    python test_audio_quality.py                    # Run full test
    python test_audio_quality.py --analyze-only     # Only analyze input audio files
    python test_audio_quality.py --help             # Show help

Requirements:
    - Python 3.9-3.11 (TTS library requirement)
    - All dependencies in requirements.txt installed
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Add the root directory to the path
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class AudioInfo:
    """Information about an audio file."""
    path: str
    name: str
    duration: float
    sample_rate: int
    is_valid: bool
    error: Optional[str] = None
    
    @property
    def is_suitable_for_cloning(self) -> bool:
        """Check if the audio file is suitable for voice cloning (3-30 seconds)."""
        return self.is_valid and 3.0 <= self.duration <= 30.0


def get_audio_info(audio_path: str) -> AudioInfo:
    """
    Get detailed information about an audio file.
    
    Args:
        audio_path: Path to the audio file.
        
    Returns:
        AudioInfo object with file details.
    """
    path = Path(audio_path)
    try:
        import librosa
        import soundfile as sf
        
        # Get duration
        duration = librosa.get_duration(path=str(path))
        
        # Get sample rate
        info = sf.info(str(path))
        sample_rate = info.samplerate
        
        return AudioInfo(
            path=str(path),
            name=path.name,
            duration=duration,
            sample_rate=sample_rate,
            is_valid=True
        )
    except Exception as e:
        return AudioInfo(
            path=str(path),
            name=path.name,
            duration=0.0,
            sample_rate=0,
            is_valid=False,
            error=str(e)
        )


def analyze_audio_files(audio_dir: Path) -> Tuple[List[AudioInfo], dict]:
    """
    Analyze all audio files in a directory.
    
    Args:
        audio_dir: Path to the directory containing audio files.
        
    Returns:
        Tuple of (list of AudioInfo objects, statistics dictionary).
    """
    audio_files = sorted(audio_dir.glob("*.wav"))
    
    infos = []
    stats = {
        "total": len(audio_files),
        "valid": 0,
        "invalid": 0,
        "too_short": 0,
        "too_long": 0,
        "suitable": 0,
        "total_duration": 0.0,
        "min_duration": float("inf"),
        "max_duration": 0.0,
    }
    
    for audio_file in audio_files:
        info = get_audio_info(str(audio_file))
        infos.append(info)
        
        if info.is_valid:
            stats["valid"] += 1
            stats["total_duration"] += info.duration
            stats["min_duration"] = min(stats["min_duration"], info.duration)
            stats["max_duration"] = max(stats["max_duration"], info.duration)
            
            if info.duration < 3.0:
                stats["too_short"] += 1
            elif info.duration > 30.0:
                stats["too_long"] += 1
            else:
                stats["suitable"] += 1
        else:
            stats["invalid"] += 1
    
    if stats["valid"] > 0:
        stats["avg_duration"] = stats["total_duration"] / stats["valid"]
    else:
        stats["avg_duration"] = 0.0
        stats["min_duration"] = 0.0
    
    return infos, stats


def print_audio_analysis(infos: List[AudioInfo], stats: dict) -> None:
    """Print a formatted analysis of the audio files."""
    print("\n" + "=" * 70)
    print("Audio File Analysis Report")
    print("=" * 70)
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Total files:     {stats['total']}")
    print(f"   Valid files:     {stats['valid']}")
    print(f"   Invalid files:   {stats['invalid']}")
    print(f"   Too short (<3s): {stats['too_short']}")
    print(f"   Too long (>30s): {stats['too_long']}")
    print(f"   Suitable (3-30s): {stats['suitable']}")
    print(f"\n⏱️  Duration Statistics:")
    print(f"   Total duration:  {stats['total_duration']:.2f}s ({stats['total_duration']/60:.2f}min)")
    print(f"   Min duration:    {stats['min_duration']:.2f}s")
    print(f"   Max duration:    {stats['max_duration']:.2f}s")
    print(f"   Avg duration:    {stats['avg_duration']:.2f}s")
    
    print("\n" + "-" * 70)
    print("Detailed File List:")
    print("-" * 70)
    
    # Group by status
    suitable = [i for i in infos if i.is_suitable_for_cloning]
    too_short = [i for i in infos if i.is_valid and i.duration < 3.0]
    too_long = [i for i in infos if i.is_valid and i.duration > 30.0]
    invalid = [i for i in infos if not i.is_valid]
    
    if suitable:
        print(f"\n✅ Suitable for voice cloning ({len(suitable)} files):")
        for info in suitable[:20]:  # Show first 20
            print(f"   {info.name}: {info.duration:.2f}s @ {info.sample_rate}Hz")
        if len(suitable) > 20:
            print(f"   ... and {len(suitable) - 20} more files")
    
    if too_short:
        print(f"\n⚠️  Too short for cloning ({len(too_short)} files):")
        for info in too_short[:10]:
            print(f"   {info.name}: {info.duration:.2f}s (minimum 3s required)")
        if len(too_short) > 10:
            print(f"   ... and {len(too_short) - 10} more files")
    
    if too_long:
        print(f"\n⚠️  Too long for cloning ({len(too_long)} files):")
        for info in too_long[:10]:
            print(f"   {info.name}: {info.duration:.2f}s (will be trimmed to 30s)")
        if len(too_long) > 10:
            print(f"   ... and {len(too_long) - 10} more files")
    
    if invalid:
        print(f"\n❌ Invalid files ({len(invalid)} files):")
        for info in invalid[:10]:
            print(f"   {info.name}: {info.error}")
        if len(invalid) > 10:
            print(f"   ... and {len(invalid) - 10} more files")


def run_voice_cloning_test(
    audio_dir: Path,
    output_dir: Path,
    test_texts: List[str],
    max_reference_files: int = 5,
    language: str = "fr"
) -> List[dict]:
    """
    Run voice cloning tests with the provided audio files.
    
    Args:
        audio_dir: Directory containing reference audio files.
        output_dir: Directory to save generated audio files.
        test_texts: List of texts to synthesize.
        max_reference_files: Maximum number of reference files to use.
        language: Language code for synthesis.
        
    Returns:
        List of test results.
    """
    from src.api import clone_voice_auto
    
    results = []
    
    print("\n" + "=" * 70)
    print("Voice Cloning Test")
    print("=" * 70)
    
    for i, text in enumerate(test_texts):
        output_path = output_dir / f"generated_{i+1:02d}.wav"
        result = {
            "test_number": i + 1,
            "text": text,
            "output_path": str(output_path),
            "success": False,
            "duration": None,
            "generation_time": None,
            "error": None
        }
        
        print(f"\n📝 Test {i+1}/{len(test_texts)}")
        print(f"   Text: \"{text}\"")
        print(f"   Output: {output_path}")
        
        start_time = time.time()
        
        try:
            generated_path = clone_voice_auto(
                audio_dir=str(audio_dir),
                text=text,
                output_path=str(output_path),
                language=language,
                whisper_model="tiny",
                use_gpu=False,
                preprocess_audio=True
            )
            
            elapsed = time.time() - start_time
            result["generation_time"] = elapsed
            
            # Verify output
            if Path(generated_path).exists():
                output_info = get_audio_info(generated_path)
                if output_info.is_valid:
                    result["success"] = True
                    result["duration"] = output_info.duration
                    print(f"   ✅ Success in {elapsed:.2f}s")
                    print(f"   📢 Output duration: {output_info.duration:.2f}s")
                else:
                    result["error"] = f"Output validation failed: {output_info.error}"
                    print(f"   ❌ Output validation failed: {output_info.error}")
            else:
                result["error"] = "Output file not created"
                print(f"   ❌ Output file not created")
                
        except Exception as e:
            elapsed = time.time() - start_time
            result["generation_time"] = elapsed
            result["error"] = str(e)
            print(f"   ❌ Error after {elapsed:.2f}s: {e}")
        
        results.append(result)
    
    return results


def print_test_summary(results: List[dict]) -> None:
    """Print a summary of the test results."""
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\n📊 Results:")
    print(f"   Total tests:    {len(results)}")
    print(f"   Successful:     {len(successful)}")
    print(f"   Failed:         {len(failed)}")
    
    if successful:
        avg_time = sum(r["generation_time"] for r in successful) / len(successful)
        avg_duration = sum(r["duration"] for r in successful) / len(successful)
        print(f"\n⏱️  Performance (successful tests):")
        print(f"   Avg generation time:  {avg_time:.2f}s")
        print(f"   Avg output duration:  {avg_duration:.2f}s")
    
    if failed:
        print(f"\n❌ Failed tests:")
        for r in failed:
            print(f"   Test {r['test_number']}: {r['error']}")


def list_generated_files(output_dir: Path) -> None:
    """List all generated audio files in the output directory."""
    print("\n" + "=" * 70)
    print("Generated Audio Files")
    print("=" * 70)
    
    generated_files = sorted(output_dir.glob("*.wav"))
    
    if generated_files:
        print(f"\n📁 Output directory: {output_dir}")
        print(f"   Files generated: {len(generated_files)}")
        print()
        for f in generated_files:
            info = get_audio_info(str(f))
            if info.is_valid:
                print(f"   📢 {f.name}: {info.duration:.2f}s @ {info.sample_rate}Hz")
            else:
                print(f"   ❌ {f.name}: {info.error}")
    else:
        print("\n   No files generated.")


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description="Test the CloneVoice program with audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze input audio files, don't run voice cloning tests"
    )
    
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="audios",
        help="Directory containing reference audio files (default: audios)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save generated audio files (default: outputs)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="fr",
        help="Language code for synthesis (default: fr)"
    )
    
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    
    # Ensure directories exist
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CloneVoice - Audio Quality Test")
    print("=" * 70)
    print(f"\n📁 Audio directory: {audio_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Analyze audio files
    infos, stats = analyze_audio_files(audio_dir)
    print_audio_analysis(infos, stats)
    
    if args.analyze_only:
        print("\n✅ Analysis complete (--analyze-only mode)")
        return
    
    # Check if we have suitable audio files
    suitable_count = stats["suitable"]
    if suitable_count == 0:
        print("\n❌ No suitable audio files found for voice cloning.")
        print("   Audio files must be between 3 and 30 seconds.")
        sys.exit(1)
    
    # Run voice cloning tests
    test_texts = [
        "Prochain arrêt, Bièvres Gare.",
        "Terminus, tout le monde descend.",
        "Attention au départ du bus.",
    ]
    
    try:
        results = run_voice_cloning_test(
            audio_dir=audio_dir,
            output_dir=output_dir,
            test_texts=test_texts,
            language=args.language
        )
        print_test_summary(results)
    except ImportError as e:
        print(f"\n⚠️  Could not import required modules: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("\n   Note: TTS library requires Python 3.9-3.11")
    
    # List generated files
    list_generated_files(output_dir)
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
