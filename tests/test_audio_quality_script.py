"""
Tests for the Audio Quality Test Script

Tests for the test_audio_quality.py module functionality.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add root directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_audio_quality import AudioInfo, get_audio_info, analyze_audio_files


class TestAudioInfo(unittest.TestCase):
    """Tests for AudioInfo dataclass."""

    def test_audio_info_creation(self):
        """Test creating an AudioInfo object."""
        info = AudioInfo(
            path="/path/to/audio.wav",
            name="audio.wav",
            duration=5.0,
            sample_rate=44100,
            is_valid=True
        )
        
        self.assertEqual(info.path, "/path/to/audio.wav")
        self.assertEqual(info.name, "audio.wav")
        self.assertEqual(info.duration, 5.0)
        self.assertEqual(info.sample_rate, 44100)
        self.assertTrue(info.is_valid)
        self.assertIsNone(info.error)

    def test_audio_info_with_error(self):
        """Test creating an AudioInfo object with an error."""
        info = AudioInfo(
            path="/path/to/audio.wav",
            name="audio.wav",
            duration=0.0,
            sample_rate=0,
            is_valid=False,
            error="File not found"
        )
        
        self.assertFalse(info.is_valid)
        self.assertEqual(info.error, "File not found")

    def test_is_suitable_for_cloning_valid(self):
        """Test is_suitable_for_cloning returns True for valid duration."""
        info = AudioInfo(
            path="/path/to/audio.wav",
            name="audio.wav",
            duration=10.0,  # Between 3 and 30 seconds
            sample_rate=44100,
            is_valid=True
        )
        
        self.assertTrue(info.is_suitable_for_cloning)

    def test_is_suitable_for_cloning_too_short(self):
        """Test is_suitable_for_cloning returns False for short audio."""
        info = AudioInfo(
            path="/path/to/audio.wav",
            name="audio.wav",
            duration=2.0,  # Less than 3 seconds
            sample_rate=44100,
            is_valid=True
        )
        
        self.assertFalse(info.is_suitable_for_cloning)

    def test_is_suitable_for_cloning_too_long(self):
        """Test is_suitable_for_cloning returns False for long audio."""
        info = AudioInfo(
            path="/path/to/audio.wav",
            name="audio.wav",
            duration=35.0,  # More than 30 seconds
            sample_rate=44100,
            is_valid=True
        )
        
        self.assertFalse(info.is_suitable_for_cloning)

    def test_is_suitable_for_cloning_invalid(self):
        """Test is_suitable_for_cloning returns False for invalid audio."""
        info = AudioInfo(
            path="/path/to/audio.wav",
            name="audio.wav",
            duration=10.0,
            sample_rate=44100,
            is_valid=False
        )
        
        self.assertFalse(info.is_suitable_for_cloning)

    def test_is_suitable_for_cloning_boundary_min(self):
        """Test is_suitable_for_cloning at minimum boundary (3s)."""
        info = AudioInfo(
            path="/path/to/audio.wav",
            name="audio.wav",
            duration=3.0,  # Exactly 3 seconds
            sample_rate=44100,
            is_valid=True
        )
        
        self.assertTrue(info.is_suitable_for_cloning)

    def test_is_suitable_for_cloning_boundary_max(self):
        """Test is_suitable_for_cloning at maximum boundary (30s)."""
        info = AudioInfo(
            path="/path/to/audio.wav",
            name="audio.wav",
            duration=30.0,  # Exactly 30 seconds
            sample_rate=44100,
            is_valid=True
        )
        
        self.assertTrue(info.is_suitable_for_cloning)


class TestGetAudioInfo(unittest.TestCase):
    """Tests for get_audio_info function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_audio_info_nonexistent_file(self):
        """Test getting audio info for a non-existent file."""
        info = get_audio_info("/path/to/nonexistent.wav")
        
        self.assertFalse(info.is_valid)
        self.assertEqual(info.duration, 0.0)
        self.assertIsNotNone(info.error)


class TestAnalyzeAudioFiles(unittest.TestCase):
    """Tests for analyze_audio_files function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('test_audio_quality.get_audio_info')
    def test_analyze_empty_directory(self, mock_get_info):
        """Test analyzing an empty directory."""
        infos, stats = analyze_audio_files(Path(self.temp_dir))
        
        self.assertEqual(len(infos), 0)
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["valid"], 0)

    @patch('test_audio_quality.get_audio_info')
    def test_analyze_mixed_files(self, mock_get_info):
        """Test analyzing a directory with mixed valid/invalid files."""
        # Create dummy .wav files
        audio1 = Path(self.temp_dir) / "audio1.wav"
        audio2 = Path(self.temp_dir) / "audio2.wav"
        audio3 = Path(self.temp_dir) / "audio3.wav"
        
        for path in [audio1, audio2, audio3]:
            path.touch()
        
        # Mock responses: valid short, valid suitable, invalid
        mock_get_info.side_effect = [
            AudioInfo(str(audio1), "audio1.wav", 2.0, 44100, True),
            AudioInfo(str(audio2), "audio2.wav", 10.0, 44100, True),
            AudioInfo(str(audio3), "audio3.wav", 0.0, 0, False, "Error"),
        ]
        
        infos, stats = analyze_audio_files(Path(self.temp_dir))
        
        self.assertEqual(len(infos), 3)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["valid"], 2)
        self.assertEqual(stats["invalid"], 1)
        self.assertEqual(stats["too_short"], 1)
        self.assertEqual(stats["suitable"], 1)


if __name__ == "__main__":
    unittest.main()
