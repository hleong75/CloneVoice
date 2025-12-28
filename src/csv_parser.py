"""
CSV Parser Module

Handles reading and validating CSV files containing audio references and transcriptions.
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class AudioEntry:
    """Represents a single entry from the CSV file."""
    audio_id: str
    transcription: str
    audio_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate the entry after initialization."""
        if not self.audio_id:
            raise ValueError("Audio ID cannot be empty")
        if not self.transcription:
            raise ValueError("Transcription cannot be empty")


class CSVParser:
    """
    Parses CSV files containing audio references and transcriptions.
    
    Expected CSV format:
    - Column 1: Audio ID/number (used to locate audio files)
    - Column 2: Transcription text
    """
    
    # Constants for audio file handling
    SUPPORTED_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    NUMERIC_ID_PADDING_LENGTHS = [2, 3, 4, 5]  # Common zero-padding lengths for numeric IDs
    
    def __init__(self, audio_directory: Optional[str] = None):
        """
        Initialize the CSV parser.
        
        Args:
            audio_directory: Directory containing audio files. If None,
                           uses the directory containing the CSV file.
        """
        self.audio_directory = audio_directory
        self.supported_extensions = self.SUPPORTED_EXTENSIONS
    
    def parse(self, csv_path: str) -> List[AudioEntry]:
        """
        Parse a CSV file and return a list of AudioEntry objects.
        
        Args:
            csv_path: Path to the CSV file.
            
        Returns:
            List of AudioEntry objects.
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If the CSV format is invalid.
        """
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Determine audio directory
        audio_dir = Path(self.audio_directory) if self.audio_directory else csv_path.parent
        
        entries = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',;|\t')
            except csv.Error:
                dialect = csv.excel  # Default to comma-separated
            
            reader = csv.reader(f, dialect)
            
            # Check if first row is header
            first_row = next(reader, None)
            if first_row is None:
                raise ValueError("CSV file is empty")
            
            # Try to determine if first row is header
            if self._is_header_row(first_row):
                pass  # Skip header
            else:
                # First row is data, process it
                entry = self._process_row(first_row, audio_dir, 1)
                if entry:
                    entries.append(entry)
            
            # Process remaining rows
            for line_num, row in enumerate(reader, start=2):
                if not row or all(cell.strip() == '' for cell in row):
                    continue  # Skip empty rows
                    
                entry = self._process_row(row, audio_dir, line_num)
                if entry:
                    entries.append(entry)
        
        if not entries:
            raise ValueError("No valid entries found in CSV file")
        
        return entries
    
    def _is_header_row(self, row: List[str]) -> bool:
        """
        Determine if a row is likely a header row.
        
        Args:
            row: List of cell values.
            
        Returns:
            True if the row appears to be a header.
        """
        if len(row) < 2:
            return False
        
        header_keywords = [
            'id', 'audio', 'number', 'num', 'no', 'n°', 'numero',
            'text', 'transcription', 'transcript', 'texte'
        ]
        
        first_cell = row[0].lower().strip()
        second_cell = row[1].lower().strip()
        
        return any(kw in first_cell or kw in second_cell for kw in header_keywords)
    
    def _process_row(self, row: List[str], audio_dir: Path, line_num: int) -> Optional[AudioEntry]:
        """
        Process a single CSV row.
        
        Args:
            row: List of cell values.
            audio_dir: Directory containing audio files.
            line_num: Line number for error reporting.
            
        Returns:
            AudioEntry object or None if row is invalid.
        """
        if len(row) < 2:
            print(f"Warning: Line {line_num} has insufficient columns, skipping")
            return None
        
        audio_id = row[0].strip()
        transcription = row[1].strip()
        
        if not audio_id or not transcription:
            print(f"Warning: Line {line_num} has empty fields, skipping")
            return None
        
        # Try to find the audio file
        audio_path = self._find_audio_file(audio_id, audio_dir)
        
        return AudioEntry(
            audio_id=audio_id,
            transcription=transcription,
            audio_path=audio_path
        )
    
    def _find_audio_file(self, audio_id: str, audio_dir: Path) -> Optional[str]:
        """
        Find an audio file matching the given ID.
        
        Args:
            audio_id: Audio identifier (can be filename with or without extension).
            audio_dir: Directory to search in.
            
        Returns:
            Full path to the audio file, or None if not found.
        """
        # First, check if audio_id is already a valid path
        if Path(audio_id).exists():
            return str(Path(audio_id).absolute())
        
        # Check if audio_id with directory is valid
        direct_path = audio_dir / audio_id
        if direct_path.exists():
            return str(direct_path.absolute())
        
        # Try adding extensions
        for ext in self.supported_extensions:
            # Try exact match
            path = audio_dir / f"{audio_id}{ext}"
            if path.exists():
                return str(path.absolute())
            
            # Try with leading zeros removed/added for numeric IDs
            if audio_id.isdigit():
                # Try without leading zeros
                path = audio_dir / f"{int(audio_id)}{ext}"
                if path.exists():
                    return str(path.absolute())
                
                # Try with common padding
                for padding in self.NUMERIC_ID_PADDING_LENGTHS:
                    padded_id = audio_id.zfill(padding)
                    path = audio_dir / f"{padded_id}{ext}"
                    if path.exists():
                        return str(path.absolute())
        
        return None


def create_csv_from_entries(entries: List[AudioEntry], output_path: str) -> None:
    """
    Create a CSV file from a list of AudioEntry objects.
    
    Args:
        entries: List of AudioEntry objects.
        output_path: Path to save the CSV file.
    """
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['audio_id', 'transcription'])
        
        for entry in entries:
            writer.writerow([entry.audio_id, entry.transcription])
