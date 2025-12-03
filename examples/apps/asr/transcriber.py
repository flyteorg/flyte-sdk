# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "nemo_toolkit[asr]",
#     "numpy",
#     "flyte>=2.0.0b29"
# ]
# ///

"""
Parakeet Multi-talker Speech Transcription Service

This module implements a GPU-accelerated speech transcription service using
NVIDIA's Parakeet multi-talker streaming model. It can be called from other
Flyte apps for real-time transcription.

Based on Modal's parakeet_multitalker.py example.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import flyte
from flyte.app import App

# NeMo imports (will be available in the container)
try:
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.models.multispeaker.sortformer_diar_asr_model import (
        SortformerEncLabelModel,
    )
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("NeMo not available - will be installed in container")


@dataclass
class TranscriptionConfig:
    """Configuration for streaming transcription."""

    speaker_cache_len: int = 188
    asr_context_size: int = 10
    asr_decoder_context_len: int = 3
    asr_batch_size: int = 4
    diar_batch_size: int = 4
    sample_rate: int = 16000
    chunk_size_frames: int = 13  # Number of frames per chunk
    bytes_per_sample: int = 2  # int16 audio


class TranscriptionService:
    """
    GPU-accelerated speech transcription service using Parakeet model.

    This service loads NVIDIA's multi-talker Parakeet model and provides
    streaming transcription capabilities.
    """

    def __init__(self):
        self.diar_model: Optional[any] = None
        self.asr_model: Optional[any] = None
        self.streaming_buffer: Optional[CacheAwareStreamingAudioBuffer] = None
        self.config = TranscriptionConfig()
        self._chunk_size_bytes = self.config.chunk_size_frames * self.config.sample_rate * self.config.bytes_per_sample

    def load_models(self):
        """Load the speaker diarization and ASR models onto GPU."""
        if not NEMO_AVAILABLE:
            raise RuntimeError("NeMo is not available. This code must run in the container.")

        print("Loading speaker diarization model...")
        self.diar_model = (
            SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
            .eval()
            .to(torch.device("cuda"))
        )

        print("Loading ASR model...")
        self.asr_model = (
            ASRModel.from_pretrained("nvidia/multitalker-parakeet-streaming-0.6b-v1").eval().to(torch.device("cuda"))
        )

        print("Initializing streaming buffer...")
        self._init_streaming_buffer()

        print("Models loaded successfully!")

    def _init_streaming_buffer(self):
        """Initialize the streaming audio buffer for chunked processing."""
        from nemo.collections.asr.parts.utils.multitask_audio_preprocessing import (
            MultitaskAudioPreprocessingConfig,
        )
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        # Create preprocessing config
        preproc_config = MultitaskAudioPreprocessingConfig(
            sample_rate=self.config.sample_rate,
            window_size_sec=0.025,
            window_stride_sec=0.01,
            features=80,
            n_fft=512,
            normalize="per_feature",
            dither=0.0,
            pad_to=0,
        )

        self.streaming_buffer = CacheAwareStreamingAudioBuffer(
            preprocessor_config=preproc_config,
            model=self.asr_model,
            cache_len=self.config.speaker_cache_len,
            chunk_len=self.config.asr_context_size,
            decoder_context_len=self.config.asr_decoder_context_len,
        )

    def transcribe_chunk(self, audio_bytes: bytes, stream_id: int = 0) -> tuple[str, bool]:
        """
        Transcribe an audio chunk.

        Args:
            audio_bytes: Raw audio bytes (int16 PCM format)
            stream_id: Unique identifier for this audio stream

        Returns:
            Tuple of (transcription_text, is_final)
        """
        if not self.diar_model or not self.asr_model:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Convert int16 bytes to float32 numpy array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Append audio to streaming buffer
        with torch.inference_mode(), torch.cuda.amp.autocast():
            _processed_signal, _processed_signal_length, stream_id = self.streaming_buffer.append_audio(
                audio_float, stream_id=stream_id
            )

            # Get next chunk to process
            result = self.streaming_buffer.get_next_chunk()

            if result is None:
                return "", False

            chunk_audio, chunk_len, _chunk_id = result

            # Perform speaker-tagged transcription
            transcription = self._perform_inference(chunk_audio, chunk_len)

            return transcription, True

    def _perform_inference(self, audio: torch.Tensor, audio_length: torch.Tensor) -> str:
        """
        Perform inference on audio chunk.

        Args:
            audio: Audio tensor
            audio_length: Length of audio

        Returns:
            Transcribed text with speaker tags
        """
        # Get speaker embeddings from diarization model
        _diar_logits, diar_embs = self.diar_model.forward_for_export(
            processed_signal=audio, processed_signal_length=audio_length
        )

        # Perform multi-speaker ASR
        hypotheses = self.asr_model.perform_parallel_streaming_stt_spk(
            audio_signal=audio,
            audio_signal_length=audio_length,
            speaker_embs=diar_embs,
            batch_size=self.config.asr_batch_size,
        )

        # Format output with speaker labels
        if hypotheses and len(hypotheses) > 0:
            # Get the best hypothesis
            hyp = hypotheses[0]
            if hasattr(hyp, "text"):
                text = hyp.text
            else:
                text = str(hyp)

            # Add speaker information if available
            if hasattr(hyp, "speaker_id"):
                return f"[Speaker {hyp.speaker_id}] {text}"
            return text

        return ""

    def reset_stream(self, stream_id: int = 0):
        """Reset the streaming buffer for a stream."""
        if self.streaming_buffer:
            self.streaming_buffer.reset_buffer(stream_id=stream_id)


# Create the Flyte app with GPU support
app = App(
    name="parakeet-transcriber",
    description="GPU-accelerated speech transcription service using Parakeet",
)

# Define the GPU image with all dependencies
gpu_image = flyte.Image.from_registry("nvcr.io/nvidia/nemo:24.07", python_version=(3, 10)).run_commands(
    "pip install --upgrade pip",
    "pip install nemo_toolkit[asr]",
)


@app.function(
    image=gpu_image,
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
        gpu=1,
        gpu_kind="A10G",  # or "T4", "A100" depending on availability
    ),
    timeout=3600,  # 1 hour timeout
    keep_warm=1,  # Keep one instance warm for low latency
)
class Transcriber:
    """
    Flyte function class for speech transcription.

    This class is deployed as a long-running GPU function that can be
    called from other Flyte apps.
    """

    def setup(self):
        """Initialize models when the function starts."""
        print("Setting up Transcriber...")
        self.service = TranscriptionService()
        self.service.load_models()
        self.active_streams = {}
        print("Transcriber ready!")

    def transcribe(self, audio_bytes: bytes, stream_id: int = 0) -> dict:
        """
        Transcribe audio bytes and return the result.

        Args:
            audio_bytes: Raw audio data (int16 PCM, 16kHz)
            stream_id: Unique stream identifier

        Returns:
            Dictionary with transcription result
        """
        try:
            text, is_final = self.service.transcribe_chunk(audio_bytes, stream_id)

            return {"success": True, "text": text, "is_final": is_final, "stream_id": stream_id}
        except Exception as e:
            return {"success": False, "error": str(e), "stream_id": stream_id}

    def reset(self, stream_id: int = 0):
        """Reset a transcription stream."""
        self.service.reset_stream(stream_id)
        return {"success": True, "message": f"Stream {stream_id} reset"}


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.DEBUG,
    )

    print("Deploying Parakeet Transcriber service...")
    deployments = flyte.deploy(app)

    if deployments:
        print("\n✅ Deployed Transcriber service:")
        for d in deployments:
            print(f"{d.table_repr()}")
    else:
        print("\n❌ Deployment failed")
