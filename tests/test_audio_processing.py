"""
Tests for Audio Processing Module

Tests for the audio preprocessing functions for voice cloning.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock numpy and scipy before importing
sys.modules['numpy'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.signal'] = MagicMock()


class TestAudioProcessingConstants(unittest.TestCase):
    """Tests for audio processing constants."""

    def test_constants_defined(self):
        """Test that constants are defined with valid values."""
        from src.audio_processing import (
            OPTIMAL_SAMPLE_RATE,
            MIN_DURATION_SECONDS,
            MAX_DURATION_SECONDS,
            OPTIMAL_DURATION_SECONDS
        )
        
        self.assertEqual(OPTIMAL_SAMPLE_RATE, 22050)
        self.assertEqual(MIN_DURATION_SECONDS, 3.0)
        self.assertEqual(MAX_DURATION_SECONDS, 30.0)
        self.assertEqual(OPTIMAL_DURATION_SECONDS, 10.0)


class TestValidateAndGetBestReference(unittest.TestCase):
    """Tests for validate_and_get_best_reference function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.audio_processing.get_audio_duration')
    def test_validates_audio_duration(self, mock_duration):
        """Test that audio duration is validated correctly."""
        from src.audio_processing import validate_and_get_best_reference
        
        # Create test audio file paths
        audio1 = os.path.join(self.temp_dir, "audio1.wav")
        audio2 = os.path.join(self.temp_dir, "audio2.wav")
        audio3 = os.path.join(self.temp_dir, "audio3.wav")
        
        # Create dummy files
        for path in [audio1, audio2, audio3]:
            Path(path).touch()
        
        # Mock durations: one too short, one optimal, one long
        mock_duration.side_effect = [2.0, 10.0, 35.0]
        
        valid_paths, messages = validate_and_get_best_reference(
            [audio1, audio2, audio3],
            min_duration=3.0,
            max_duration=30.0
        )
        
        # Only audio2 and audio3 should be valid (audio3 will be trimmed)
        self.assertEqual(len(valid_paths), 2)
        self.assertEqual(len(messages), 3)

    @patch('src.audio_processing.get_audio_duration')
    def test_sorts_by_optimal_duration(self, mock_duration):
        """Test that audio files are sorted by distance from optimal duration."""
        from src.audio_processing import validate_and_get_best_reference
        
        audio1 = os.path.join(self.temp_dir, "audio1.wav")
        audio2 = os.path.join(self.temp_dir, "audio2.wav")
        
        for path in [audio1, audio2]:
            Path(path).touch()
        
        # audio2 (10s) is closer to optimal (10s) than audio1 (20s)
        mock_duration.side_effect = [20.0, 10.0]
        
        valid_paths, _ = validate_and_get_best_reference(
            [audio1, audio2],
            min_duration=3.0,
            max_duration=30.0
        )
        
        # audio2 should come first (closer to optimal 10s)
        self.assertEqual(valid_paths[0], audio2)
        self.assertEqual(valid_paths[1], audio1)

    @patch('src.audio_processing.get_audio_duration')
    def test_handles_errors_gracefully(self, mock_duration):
        """Test that errors during validation are handled gracefully."""
        from src.audio_processing import validate_and_get_best_reference
        
        audio1 = os.path.join(self.temp_dir, "audio1.wav")
        audio2 = os.path.join(self.temp_dir, "audio2.wav")
        
        for path in [audio1, audio2]:
            Path(path).touch()
        
        # First audio raises error, second is valid
        mock_duration.side_effect = [Exception("Test error"), 10.0]
        
        valid_paths, messages = validate_and_get_best_reference(
            [audio1, audio2],
            min_duration=3.0,
            max_duration=30.0
        )
        
        # Only audio2 should be valid
        self.assertEqual(len(valid_paths), 1)
        self.assertEqual(valid_paths[0], audio2)


if __name__ == "__main__":
    unittest.main()
