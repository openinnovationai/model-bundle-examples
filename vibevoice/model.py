import base64
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

TMP_DIR = Path("/tmp")
MODEL_DIR = TMP_DIR / "hf_model"
OUTPUT_DIR = TMP_DIR / "output"
VOICE_SAMPLE_PATH = Path(__file__).parent / "data" / "en-Alice_woman.wav"

logger = logging.getLogger("ray.serve")


class Model:
    """VibeVoice-1.5B text-to-speech model bundle.

    This model provides text-to-speech functionality using the community
    vibevoice repo (https://github.com/vibevoice-community/VibeVoice) and
    Microsoft's VibeVoice-1.5B weights from Hugging Face
    (https://huggingface.co/microsoft/VibeVoice-1.5B).

    This example only works for a single speaker. It returns the audio as a
    base64-encoded string. A decoder must be used to convert the string to an
    audio waveform.
    """

    def __init__(self, data_dir: str, config: dict[str, Any]):
        """Initialize the model."""

        self.model = None
        self.processor = None
        self.device = None
        self.data_dir = data_dir
        self.config = config

    def _download_model(self, model_dir: Path) -> str:
        """Download the model from HF into /tmp."""

        snapshot_download(
            repo_id="microsoft/VibeVoice-1.5B",
            local_dir=model_dir,
        )

        return model_dir

    def _wav_to_base64(self, wav_path: Path) -> str:
        """Convert a WAV file to base64."""

        with open(wav_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def load(self):
        """Load the VibeVoice model into GPU memory and warm up."""

        logger.info("Loading model...")

        # Init dirs
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Download the model from HF
        model_path = self._download_model(MODEL_DIR)

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Load model
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # Move to device if not using device_map
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)

        # Set to evaluation mode
        self.model.eval()

        # Warm up tokenizer
        test_inputs = self.processor(
            text=["Speaker 1: Hi."],
            voice_samples=[[str(VOICE_SAMPLE_PATH)]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Warm up model
        self.model.set_ddpm_inference_steps(num_steps=2)
        self.model.generate(
            **test_inputs,
            max_new_tokens=None,
            cfg_scale=1.3,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
        )

        # Set to deployment inference steps
        self.model.set_ddpm_inference_steps(num_steps=10)
        logger.info("Model loaded successfully!")

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Generate speech audio from text input. Handles only single speaker.

        Args:
            inputs (dict[str, Any]): Dictionary containing:
                - text (str): Input text to convert to speech

        Returns:
            dict[str, Any]: Dictionary containing:
                - audio: Base64-encoded audio waveform
                - input_text: The original input text
                - device_used: Device used for inference
                - time_taken: Time taken for inference
        """

        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        # Extract inputs
        text = inputs.get("text", "")
        if not text:
            raise ValueError("Input text is required.")

        logger.info(
            f"Generating speech for text: '{text[:50]}{'...' if len(text) > 50 else ''}'"
        )

        # Process input text with speaker information
        processed_inputs = self.processor(
            text=[text],
            voice_samples=[[str(VOICE_SAMPLE_PATH)]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Convert lists to tensors and move tensor inputs to device
        for k, v in processed_inputs.items():
            if isinstance(v, torch.Tensor):
                processed_inputs[k] = v.to(self.device)

        # Generate speech
        time_start = time.time()
        generated_outputs = self.model.generate(
            **processed_inputs,
            max_new_tokens=None,  # Value taken from source.
            cfg_scale=1.3,  # Value taken from source.
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},  # Value taken from source.
        )
        time_end = time.time()

        # Decode to audio waveform
        audio_filepath = OUTPUT_DIR / f"{uuid.uuid4()}.wav"
        self.processor.save_audio(
            generated_outputs.speech_outputs[0],  # First (and only) batch item
            output_path=str(audio_filepath),
        )

        # Convert to base64 and cleanup to save space
        audio_base64 = self._wav_to_base64(audio_filepath)
        audio_filepath.unlink()

        return {
            "audio": audio_base64,
            "input_text": text,
            "device_used": str(self.device),
            "time_taken": time_end - time_start,
        }
