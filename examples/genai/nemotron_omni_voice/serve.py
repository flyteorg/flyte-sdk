"""
Low-latency voice chat backed by NVIDIA Nemotron-3-Nano-Omni-30B-A3B.

The Omni model is a multimodal (audio / video / image / text -> text) variant of
Nemotron-3-Nano. It natively ingests audio, so the client streams raw microphone
audio straight to the model -- there is no separate speech-to-text stage. The
server emits text tokens; the client speaks them with a local TTS engine.

    mic ──► [client] capture audio ──► (base64 WAV in an OpenAI chat message)
                                          │
                                          ▼
                        [Flyte app] vLLM serving Nemotron-3-Nano-Omni-30B-A3B
                                          │  (audio-in → text-out, streamed)
                                          ▼
                          [client] sentence-by-sentence TTS ──► speaker

This file is the *server*. See ``voice_client.py`` for the microphone client.

Everything here mirrors ``examples/genai/vllm/vllm_app.py``: we prefetch the model
into the Flyte object store once, then serve it with the ``flyteplugins-vllm``
``VLLMAppEnvironment``, which wraps ``vllm serve`` and exposes an OpenAI-compatible
API at ``<app-url>/v1``.

Deploy
------
1. Create an HF token secret (the Nemotron weights are gated by license):

       flyte create secret HF_TOKEN <your-hf-token>

2. Prefetch the weights into the object store and deploy the app:

       python examples/genai/nemotron_omni_voice/serve.py

   or, to deploy against an already-prefetched model, use the CLI:

       flyte deploy examples/genai/nemotron_omni_voice/serve.py vllm_app

3. Point the client at the printed app URL:

       python examples/genai/nemotron_omni_voice/voice_client.py --endpoint <app-url>
"""

from flyteplugins.vllm import VLLMAppEnvironment

import flyte
import flyte.app

# ---------------------------------------------------------------------------
# Model
#
# The FP8 checkpoint is ~33 GB and fits on a single 48 GB L40S with room left
# for the KV cache and the audio/vision encoders. (BF16 would need ~64 GB and
# would not fit on one L40S.)
# ---------------------------------------------------------------------------

MODEL_REPO = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
MODEL_ID = "nemotron-omni"  # the name clients pass as `model=...`

# ---------------------------------------------------------------------------
# Image
#
# The Omni model needs vLLM >= 0.20 and trust-remote-code. We mirror the
# plugin's default image (flashinfer for fast FP8 attention on Ada/L40S) but
# pin a newer vLLM and add the audio decoding libraries vLLM uses to read the
# incoming wav/mp3 payloads.
# ---------------------------------------------------------------------------

VLLM_VERSION = "0.20.0"

image = (
    flyte.Image.from_debian_base(name="nemotron-omni-vllm", install_flyte=False)
    .with_pip_packages("flashinfer-python", "flashinfer-cubin")
    .with_pip_packages("flashinfer-jit-cache", index_url="https://flashinfer.ai/whl/cu129")
    .with_pip_packages("flyteplugins-vllm", pre=True)
    # vLLM goes in its own layer (dependency conflict with flyte on protovalidate).
    .with_pip_packages(f"vllm=={VLLM_VERSION}", "transformers>=4.57.0")
    # Audio decoding for the multimodal input pipeline.
    .with_pip_packages("librosa", "soundfile")
)

# ---------------------------------------------------------------------------
# Serving app
#
# `extra_args` is passed verbatim to `vllm serve`. The flags below come from the
# NVIDIA Omni vLLM recipe, retuned for *latency* on a single user rather than
# throughput:
#
#   --max-model-len 32768   smaller context -> smaller KV cache, fits 48 GB,
#                           and shorter sequences mean lower per-token latency.
#                           (The model supports up to 131072; raise if you need
#                           long audio context and have the memory.)
#   --max-num-seqs 4        a voice session is ~1 concurrent request; a small
#                           batch ceiling minimizes scheduling/queueing latency.
#   --kv-cache-dtype fp8    halves KV-cache memory and bandwidth -> faster decode.
#   --async-scheduling      recommended by NVIDIA for best per-request latency.
#   --reasoning-parser nemotron_v3   required to parse this checkpoint correctly.
#                           Note: the client disables <think> (chat_template_kwargs
#                           enable_thinking=false) so we don't pay reasoning
#                           latency on a live conversation.
# ---------------------------------------------------------------------------

vllm_app = VLLMAppEnvironment(
    name="nemotron-omni-voice",
    model_id=MODEL_ID,
    # Filled in at deploy time from the prefetch run (see __main__). Using a
    # RunOutput means the app always materializes the prefetched weights from
    # the object store instead of re-downloading from HF on every replica.
    model_hf_path=MODEL_REPO,  # overridden with model_path in __main__
    image=image,
    resources=flyte.Resources(cpu="8", memory="32Gi", gpu="L40s:1", disk="80Gi"),
    # Keep at least one warm replica: cold-starting a 33 GB model defeats the
    # entire point of a latency-optimized voice app. Set min=0 + scaledown_after
    # if you'd rather save GPU cost and can tolerate a cold start.
    scaling=flyte.app.Scaling(replicas=(1, 1)),
    # Download weights to local disk then load. The streaming loader
    # (stream_model=True) gives faster cold starts but is not guaranteed for
    # multimodal FP8 checkpoints, so we use the robust path here.
    stream_model=False,
    requires_auth=False,  # demo only; enable auth for anything real.
    extra_args=[
        "--trust-remote-code",
        "--reasoning-parser",
        "nemotron_v3",
        "--kv-cache-dtype",
        "fp8",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.90",
        "--async-scheduling",
    ],
)


if __name__ == "__main__":
    from flyte.remote import Run

    flyte.init_from_config()

    # Step 1: Prefetch the model into the Flyte object store (once). Subsequent
    # deploys reuse the cached run output -- no repeated multi-GB HF download.
    run: Run = flyte.prefetch.hf_model(
        repo=MODEL_REPO,
        modality=("audio", "video", "image", "text"),
        hf_token_key="HF_TOKEN",
        resources=flyte.Resources(cpu="4", memory="16Gi", disk="80Gi"),
    )
    run.wait()
    print(f"Prefetched model: {run.url}")

    # Step 2: Deploy, serving the prefetched weights from the object store.
    app = flyte.serve(
        vllm_app.clone_with(
            name=vllm_app.name,
            model_id=MODEL_ID,
            model_hf_path=None,
            model_path=flyte.app.RunOutput(type="directory", run_name=run.name),
        )
    )
    print(f"Deployed Nemotron-Omni voice app: {app.url}")
    print(f"OpenAI-compatible endpoint: {app.url}/v1")
    print("\nStart talking:")
    print(f"  python voice_client.py --endpoint {app.url} --model {MODEL_ID}")
