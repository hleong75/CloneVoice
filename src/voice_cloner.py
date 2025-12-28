"""
Voice Cloning Module

Core module that handles voice cloning using state-of-the-art AI models.
Uses Coqui TTS with XTTS v2 for high-quality voice cloning.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import torch
import numpy as np
from tqdm import tqdm


class VoiceCloner:
    """
    Voice cloning engine using XTTS v2 model.
    
    XTTS v2 is a state-of-the-art voice cloning model that can:
    - Clone voices from short audio samples (as short as 6 seconds)
    - Generate natural-sounding speech in multiple languages
    - Preserve the speaker's characteristics including accent and prosody
    """
    
    # XTTS v2 supported languages
    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
    ]
    
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the voice cloning engine.
        
        Args:
            model_name: Name of the TTS model to use.
            device: Specific device to use ('cuda', 'cpu', etc.).
            use_gpu: If True and CUDA is available, use GPU.
        """
        self.model_name = model_name
        self.tts = None
        self.device = device
        self.use_gpu = use_gpu
        self._initialized = False
        
    def _initialize(self) -> None:
        """Lazy initialization of the TTS model."""
        if self._initialized:
            return
            
        print("Initializing voice cloning model... This may take a moment on first run.")
        
        # Register TTS configuration classes as safe globals for PyTorch 2.6+
        # This is required because torch.load() now defaults to weights_only=True
        self._register_tts_safe_globals()
        
        # Patch GPT2InferenceModel to support transformers >= 4.50
        # This is required because GenerationMixin is no longer inherited by default
        self._patch_gpt2_inference_model()
        
        from TTS.api import TTS
        
        # Determine device
        if self.device:
            device = self.device
        elif self.use_gpu and torch.cuda.is_available():
            device = "cuda"
            print("Using GPU for voice cloning")
        else:
            device = "cpu"
            print("Using CPU for voice cloning (this may be slower)")
        
        # Initialize TTS model
        self.tts = TTS(model_name=self.model_name).to(device)
        self._initialized = True
        
        print("Voice cloning model initialized successfully!")
    
    def _register_tts_safe_globals(self) -> None:
        """
        Register TTS configuration classes as safe globals for torch.load().
        
        In PyTorch 2.6+, torch.load() defaults to weights_only=True for security.
        TTS models use custom configuration classes that need to be explicitly
        allowed for safe loading.
        """
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            torch.serialization.add_safe_globals([XttsConfig])
        except ImportError:
            # TTS package may not have this config in all versions
            pass
        
        try:
            from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
            torch.serialization.add_safe_globals([XttsArgs, XttsAudioConfig])
        except ImportError:
            pass
        
        try:
            from TTS.config import BaseAudioConfig, BaseDatasetConfig
            torch.serialization.add_safe_globals([BaseAudioConfig, BaseDatasetConfig])
        except ImportError:
            pass
    
    def _patch_gpt2_inference_model(self) -> None:
        """
        Patch GPT2InferenceModel to inherit from GenerationMixin.
        
        In transformers >= 4.50, PreTrainedModel no longer inherits from
        GenerationMixin by default. The XTTS v2 model uses GPT2InferenceModel
        which needs GenerationMixin to call the generate() method.
        
        This patch dynamically adds GenerationMixin to the class bases.
        """
        try:
            from transformers import GenerationMixin
            from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel
            
            # Check if GenerationMixin is already in the bases
            if GenerationMixin not in GPT2InferenceModel.__bases__:
                # Add GenerationMixin to the class bases
                GPT2InferenceModel.__bases__ = (
                    GPT2InferenceModel.__bases__ + (GenerationMixin,)
                )
        except ImportError:
            # TTS package may not have this module in all versions
            pass
    
    def clone_voice(
        self,
        text: str,
        reference_audio: Union[str, List[str]],
        output_path: str,
        language: str = "fr"
    ) -> str:
        """
        Clone a voice and generate speech.
        
        Args:
            text: Text to synthesize.
            reference_audio: Path to reference audio file(s) for voice cloning.
            output_path: Path to save the generated audio.
            language: Target language code (e.g., 'en', 'fr', 'es', 'de').
            
        Returns:
            Path to the generated audio file.
        """
        self._initialize()
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle multiple reference audios
        if isinstance(reference_audio, list):
            speaker_wav = reference_audio
        else:
            speaker_wav = reference_audio
        
        # Generate speech with cloned voice
        self.tts.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker_wav=speaker_wav,
            language=language
        )
        
        return str(output_path.absolute())
    
    def batch_clone(
        self,
        texts: List[str],
        reference_audio: Union[str, List[str]],
        output_dir: str,
        language: str = "fr",
        filename_prefix: str = "cloned"
    ) -> List[str]:
        """
        Generate multiple audio files with the cloned voice.
        
        Args:
            texts: List of texts to synthesize.
            reference_audio: Path to reference audio file(s).
            output_dir: Directory to save generated audio files.
            language: Target language code.
            filename_prefix: Prefix for output filenames.
            
        Returns:
            List of paths to generated audio files.
        """
        self._initialize()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        
        for i, text in enumerate(tqdm(texts, desc="Generating audio")):
            output_path = output_dir / f"{filename_prefix}_{i+1:04d}.wav"
            
            try:
                result = self.clone_voice(
                    text=text,
                    reference_audio=reference_audio,
                    output_path=str(output_path),
                    language=language
                )
                output_files.append(result)
                
            except Exception as e:
                print(f"Error generating audio {i+1}: {e}")
                continue
        
        return output_files
    
    def get_available_languages(self) -> List[str]:
        """
        Get list of languages supported by the model.
        
        Returns:
            List of language codes.
        """
        return self.SUPPORTED_LANGUAGES.copy()


def create_voice_cloner(use_gpu: bool = True) -> VoiceCloner:
    """
    Factory function to create a VoiceCloner instance.
    
    Args:
        use_gpu: If True and CUDA is available, use GPU.
        
    Returns:
        Configured VoiceCloner instance.
    """
    return VoiceCloner(use_gpu=use_gpu)
