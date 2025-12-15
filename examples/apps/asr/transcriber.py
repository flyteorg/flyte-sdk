# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch[gpu]",
#     "nemo_toolkit[asr]",
#     "numpy",
#     "flyte>=2.0.0b35"
# ]
# ///

"""
Parakeet Multi-talker Speech Transcription Service

This module implements a GPU-accelerated speech transcription service using
NVIDIA's Parakeet multi-talker streaming model. It can be called from other
Flyte apps for real-time transcription.

"""

import logging
import pathlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import flyte
from flyte.app.extras import FastAPIAppEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.models.multispeaker.sortformer_diar_asr_model import (
        SortformerEncLabelModel,
    )
    from nemo.collections.asr.parts.utils.multitask_audio_preprocessing import (
        MultitaskAudioPreprocessingConfig,
    )
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logger.warning("NeMo not available - will be installed in container")

_GPU = "T4"
GPU = flyte.GPU("T4", 1)
if _GPU == "T4":
    GPU = flyte.GPU("T4", 1)
    ASR_BATCH_SIZE = 2
    DIAR_BATCH_SIZE = 2
elif _GPU == "A10G":
    GPU = flyte.GPU("A10G", 1)
    ASR_BATCH_SIZE = 4
    DIAR_BATCH_SIZE = 4


@dataclass
class TranscriptionConfig:
    """Configuration for streaming transcription."""

    speaker_cache_len: int = 188
    asr_context_size: int = 10
    asr_decoder_context_len: int = 3
    asr_batch_size: int = ASR_BATCH_SIZE
    diar_batch_size: int = DIAR_BATCH_SIZE
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

        logger.info("Loading speaker diarization model...")
        self.diar_model = (
            SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
            .eval()
            .to(torch.device("cuda"))
        )

        logger.info("Loading ASR model...")
        self.asr_model = (
            ASRModel.from_pretrained("nvidia/multitalker-parakeet-streaming-0.6b-v1").eval().to(torch.device("cuda"))
        )

        logger.info("Initializing streaming buffer...")
        self._init_streaming_buffer()

        logger.info("Models loaded successfully!")

    def _init_streaming_buffer(self):
        """Initialize the streaming audio buffer for chunked processing."""
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


# Global service instance
transcription_service: Optional[TranscriptionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle with lifespan context manager."""
    global transcription_service

    # Startup
    logger.info("Starting up transcription service...")
    try:
        transcription_service = TranscriptionService()
        transcription_service.load_models()
        logger.info("Service startup complete!")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down transcription service...")
    transcription_service = None


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Parakeet Transcription Service",
    description="GPU-accelerated speech transcription service using NVIDIA NeMo Parakeet",
    version="1.0.0",
    lifespan=lifespan,
)

# Create FastAPI environment for Flyte
env = FastAPIAppEnvironment(
    name="parakeet-transcriber",
    app=app,
    description="GPU-accelerated speech transcription service using Parakeet",
    image=flyte.Image.from_debian_base(name="parakeet", python_version=(3, 11))
    # Install system dependencies first
    .with_apt_packages(
        "libsndfile1",
        "ffmpeg",
        "libsndfile1-dev",
        "git",
    )
    # Install PyTorch with CUDA support (correct way)
    .with_pip_packages(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .with_pip_packages("nemo_toolkit[asr]")
    .with_pip_packages(
        "fastapi",
        "uvicorn",
        "python-multipart",
    )
    .with_pip_packages("flyte", pre=True),
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
        gpu=GPU,
    ),
    requires_auth=False,
)


# Pydantic models for request/response
class TranscriptionRequest(BaseModel):
    stream_id: int = 0


class TranscriptionResponse(BaseModel):
    success: bool
    text: str = ""
    is_final: bool = False
    stream_id: int = 0
    error: Optional[str] = None


class ResetResponse(BaseModel):
    success: bool
    message: str
    stream_id: int = 0


@env.app.get("/health")
async def health_check():
    """Health check endpoint."""
    if transcription_service is None or transcription_service.asr_model is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": True,
    }


@env.app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
        audio: UploadFile = File(...),
        stream_id: int = 0,
):
    """
    Transcribe audio chunk.

    Args:
        audio: Audio file (int16 PCM, 16kHz)
        stream_id: Unique stream identifier for continuous transcription

    Returns:
        Transcription result with speaker information
    """
    if transcription_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Read audio bytes
        audio_bytes = await audio.read()

        # Transcribe
        text, is_final = transcription_service.transcribe_chunk(audio_bytes, stream_id)

        return TranscriptionResponse(
            success=True,
            text=text,
            is_final=is_final,
            stream_id=stream_id,
        )

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return TranscriptionResponse(
            success=False,
            error=str(e),
            stream_id=stream_id,
        )


@env.app.post("/transcribe/bytes", response_model=TranscriptionResponse)
async def transcribe_raw_bytes(
        audio_bytes: bytes,
        stream_id: int = 0,
):
    """
    Transcribe raw audio bytes (alternative endpoint for direct byte upload).

    Args:
        audio_bytes: Raw audio data (int16 PCM, 16kHz)
        stream_id: Unique stream identifier

    Returns:
        Transcription result
    """
    if transcription_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        text, is_final = transcription_service.transcribe_chunk(audio_bytes, stream_id)

        return TranscriptionResponse(
            success=True,
            text=text,
            is_final=is_final,
            stream_id=stream_id,
        )

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return TranscriptionResponse(
            success=False,
            error=str(e),
            stream_id=stream_id,
        )


@env.app.post("/reset/{stream_id}", response_model=ResetResponse)
async def reset_stream(stream_id: int):
    """
    Reset a transcription stream.

    Args:
        stream_id: Stream identifier to reset

    Returns:
        Success status
    """
    if transcription_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        transcription_service.reset_stream(stream_id)
        return ResetResponse(
            success=True,
            message=f"Stream {stream_id} reset successfully",
            stream_id=stream_id,
        )
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@env.app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Parakeet Transcription Service",
        "version": "1.0.0",
        "description": "GPU-accelerated streaming transcription using NVIDIA NeMo",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe",
            "transcribe_bytes": "/transcribe/bytes",
            "reset": "/reset/{stream_id}",
        },
    }


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.INFO,
    )

    print("🚀 Starting Parakeet Transcription Service...")
    print("=" * 60)
    print("\n✅ GPU transcription service with NVIDIA NeMo Parakeet")
    print("\n🔍 Available API Endpoints:")
    print("   GET  /health              - Health check")
    print("   POST /transcribe          - Transcribe audio file")
    print("   POST /transcribe/bytes    - Transcribe raw bytes")
    print("   POST /reset/{stream_id}   - Reset stream buffer")
    print("\nStarting server...\n")

    # Serve the transcription service
    served_app = flyte.serve(env)
    print(f"{served_app.url} - connect at {served_app.endpoint}")
