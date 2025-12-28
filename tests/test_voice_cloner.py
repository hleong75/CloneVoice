"""
Tests for VoiceCloner module

Tests for the voice cloning functionality and compatibility patches.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock torch and dependencies before importing VoiceCloner
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['tqdm'] = MagicMock()

from src.voice_cloner import VoiceCloner


class TestVoiceClonerPatches(unittest.TestCase):
    """Tests for VoiceCloner compatibility patches."""

    def test_patch_gpt2_inference_model_adds_generation_mixin(self):
        """Test that _patch_gpt2_inference_model adds GenerationMixin to bases."""
        # Create mock classes with proper inheritance
        class MockBase:
            pass
        
        class MockGenerationMixin:
            pass
        
        class MockGPT2InferenceModel(MockBase):
            pass
        
        # Patch imports
        with patch.dict('sys.modules', {
            'transformers': MagicMock(GenerationMixin=MockGenerationMixin),
            'TTS': MagicMock(),
            'TTS.tts': MagicMock(),
            'TTS.tts.layers': MagicMock(),
            'TTS.tts.layers.xtts': MagicMock(),
            'TTS.tts.layers.xtts.gpt_inference': MagicMock(
                GPT2InferenceModel=MockGPT2InferenceModel
            ),
        }):
            cloner = VoiceCloner()
            
            # Verify MockGenerationMixin is not in bases initially
            self.assertNotIn(MockGenerationMixin, MockGPT2InferenceModel.__bases__)
            
            # Run the patch
            cloner._patch_gpt2_inference_model()
            
            # Verify MockGenerationMixin was added to bases
            self.assertIn(MockGenerationMixin, MockGPT2InferenceModel.__bases__)

    def test_patch_gpt2_inference_model_handles_import_error(self):
        """Test that _patch_gpt2_inference_model handles ImportError gracefully."""
        cloner = VoiceCloner()
        
        # Patch the import to raise ImportError
        with patch.dict('sys.modules', {
            'transformers': None,
            'TTS.tts.layers.xtts.gpt_inference': None,
        }):
            # Should not raise an exception
            try:
                cloner._patch_gpt2_inference_model()
            except Exception as e:
                self.fail(f"_patch_gpt2_inference_model raised {type(e).__name__}: {e}")

    def test_patch_gpt2_inference_model_skips_if_already_patched(self):
        """Test that _patch_gpt2_inference_model doesn't duplicate GenerationMixin."""
        class MockBase:
            pass
        
        class MockGenerationMixin:
            pass
        
        # Create a class that already has MockGenerationMixin in bases
        class MockGPT2InferenceModel(MockBase, MockGenerationMixin):
            pass
        
        original_bases = MockGPT2InferenceModel.__bases__
        
        with patch.dict('sys.modules', {
            'transformers': MagicMock(GenerationMixin=MockGenerationMixin),
            'TTS': MagicMock(),
            'TTS.tts': MagicMock(),
            'TTS.tts.layers': MagicMock(),
            'TTS.tts.layers.xtts': MagicMock(),
            'TTS.tts.layers.xtts.gpt_inference': MagicMock(
                GPT2InferenceModel=MockGPT2InferenceModel
            ),
        }):
            cloner = VoiceCloner()
            cloner._patch_gpt2_inference_model()
            
            # Verify bases weren't modified (no duplicate MockGenerationMixin)
            self.assertEqual(MockGPT2InferenceModel.__bases__, original_bases)


class TestVoiceClonerInit(unittest.TestCase):
    """Tests for VoiceCloner initialization."""

    def test_init_default_params(self):
        """Test VoiceCloner initialization with default parameters."""
        cloner = VoiceCloner()
        
        self.assertEqual(cloner.model_name, "tts_models/multilingual/multi-dataset/xtts_v2")
        self.assertIsNone(cloner.device)
        self.assertTrue(cloner.use_gpu)
        self.assertTrue(cloner.preprocess_audio)
        self.assertFalse(cloner._initialized)
        
        # Check default inference parameters
        self.assertIn("temperature", cloner.inference_params)
        self.assertIn("repetition_penalty", cloner.inference_params)
        self.assertIn("speed", cloner.inference_params)

    def test_init_custom_params(self):
        """Test VoiceCloner initialization with custom parameters."""
        custom_inference_params = {"temperature": 0.5, "speed": 1.2}
        cloner = VoiceCloner(
            model_name="custom_model",
            device="cpu",
            use_gpu=False,
            preprocess_audio=False,
            inference_params=custom_inference_params
        )
        
        self.assertEqual(cloner.model_name, "custom_model")
        self.assertEqual(cloner.device, "cpu")
        self.assertFalse(cloner.use_gpu)
        self.assertFalse(cloner.preprocess_audio)
        
        # Check that custom inference params override defaults
        self.assertEqual(cloner.inference_params["temperature"], 0.5)
        self.assertEqual(cloner.inference_params["speed"], 1.2)

    def test_get_available_languages(self):
        """Test getting available languages."""
        cloner = VoiceCloner()
        languages = cloner.get_available_languages()
        
        self.assertIsInstance(languages, list)
        self.assertIn("en", languages)
        self.assertIn("fr", languages)
        self.assertIn("de", languages)

    def test_cleanup(self):
        """Test cleanup method."""
        cloner = VoiceCloner()
        # Add some fake temp dirs
        cloner._temp_dirs = ["/tmp/fake_dir_1", "/tmp/fake_dir_2"]
        
        # Cleanup should clear the list even if dirs don't exist
        cloner.cleanup()
        
        self.assertEqual(cloner._temp_dirs, [])


if __name__ == "__main__":
    unittest.main()
