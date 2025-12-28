"""
Tests for CloneVoice

These tests validate the core functionality of the voice cloning application.
"""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.csv_parser import CSVParser, AudioEntry, create_csv_from_entries


class TestAudioEntry(unittest.TestCase):
    """Tests for AudioEntry dataclass."""
    
    def test_valid_entry(self):
        """Test creating a valid audio entry."""
        entry = AudioEntry(audio_id="001", transcription="Hello world")
        self.assertEqual(entry.audio_id, "001")
        self.assertEqual(entry.transcription, "Hello world")
        self.assertIsNone(entry.audio_path)
    
    def test_entry_with_path(self):
        """Test creating an entry with audio path."""
        entry = AudioEntry(
            audio_id="001",
            transcription="Hello",
            audio_path="/path/to/audio.wav"
        )
        self.assertEqual(entry.audio_path, "/path/to/audio.wav")
    
    def test_empty_audio_id_raises(self):
        """Test that empty audio ID raises ValueError."""
        with self.assertRaises(ValueError):
            AudioEntry(audio_id="", transcription="Hello")
    
    def test_empty_transcription_raises(self):
        """Test that empty transcription raises ValueError."""
        with self.assertRaises(ValueError):
            AudioEntry(audio_id="001", transcription="")


class TestCSVParser(unittest.TestCase):
    """Tests for CSVParser class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.parser = CSVParser(audio_directory=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_csv(self, content: str, filename: str = "test.csv") -> str:
        """Helper to create a temporary CSV file."""
        path = os.path.join(self.temp_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path
    
    def _create_audio_file(self, name: str) -> str:
        """Helper to create a dummy audio file."""
        path = os.path.join(self.temp_dir, name)
        with open(path, 'w') as f:
            f.write("dummy audio content")
        return path
    
    def test_parse_simple_csv(self):
        """Test parsing a simple CSV file."""
        csv_content = "001,Hello world\n002,Goodbye world\n"
        csv_path = self._create_csv(csv_content)
        
        entries = self.parser.parse(csv_path)
        
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].audio_id, "001")
        self.assertEqual(entries[0].transcription, "Hello world")
        self.assertEqual(entries[1].audio_id, "002")
        self.assertEqual(entries[1].transcription, "Goodbye world")
    
    def test_parse_csv_with_header(self):
        """Test parsing CSV with header row."""
        csv_content = "audio_id,transcription\n001,Hello world\n"
        csv_path = self._create_csv(csv_content)
        
        entries = self.parser.parse(csv_path)
        
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].audio_id, "001")
    
    def test_parse_csv_finds_audio_files(self):
        """Test that parser finds matching audio files."""
        # Create audio file
        self._create_audio_file("001.wav")
        
        csv_content = "001,Hello world\n"
        csv_path = self._create_csv(csv_content)
        
        entries = self.parser.parse(csv_path)
        
        self.assertEqual(len(entries), 1)
        self.assertIsNotNone(entries[0].audio_path)
        self.assertTrue(entries[0].audio_path.endswith("001.wav"))
    
    def test_parse_csv_skips_empty_rows(self):
        """Test that empty rows are skipped."""
        csv_content = "001,Hello\n\n002,World\n"
        csv_path = self._create_csv(csv_content)
        
        entries = self.parser.parse(csv_path)
        
        self.assertEqual(len(entries), 2)
    
    def test_parse_nonexistent_file_raises(self):
        """Test that parsing non-existent file raises error."""
        with self.assertRaises(FileNotFoundError):
            self.parser.parse("/nonexistent/path.csv")
    
    def test_parse_empty_csv_raises(self):
        """Test that parsing empty CSV raises error."""
        csv_path = self._create_csv("")
        
        with self.assertRaises(ValueError):
            self.parser.parse(csv_path)
    
    def test_semicolon_delimiter(self):
        """Test parsing CSV with semicolon delimiter."""
        csv_content = "001;Hello world\n002;Goodbye world\n"
        csv_path = self._create_csv(csv_content)
        
        entries = self.parser.parse(csv_path)
        
        self.assertEqual(len(entries), 2)
    
    def test_tab_delimiter(self):
        """Test parsing CSV with tab delimiter."""
        csv_content = "001\tHello world\n002\tGoodbye world\n"
        csv_path = self._create_csv(csv_content)
        
        entries = self.parser.parse(csv_path)
        
        self.assertEqual(len(entries), 2)


class TestCreateCSVFromEntries(unittest.TestCase):
    """Tests for create_csv_from_entries function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_csv(self):
        """Test creating CSV from entries."""
        entries = [
            AudioEntry(audio_id="001", transcription="Hello"),
            AudioEntry(audio_id="002", transcription="World")
        ]
        
        output_path = os.path.join(self.temp_dir, "output.csv")
        create_csv_from_entries(entries, output_path)
        
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 3)  # Header + 2 entries
        self.assertEqual(rows[0], ['audio_id', 'transcription'])
        self.assertEqual(rows[1], ['001', 'Hello'])
        self.assertEqual(rows[2], ['002', 'World'])


if __name__ == "__main__":
    unittest.main()
