"""
Text Embeddings Inference (TEI) app.

Wraps Hugging Face's pre-built `text-embeddings-inference` container
(https://github.com/huggingface/text-embeddings-inference) in a Flyte
`AppEnvironment`. TEI is a high-throughput server for embedding, re-ranking,
and sequence-classification models — no Python code of our own runs in the
container, we just launch the `text-embeddings-router` binary the image ships.

Two environments are defined:

  * ``tei_cpu_env``   — CPU image, fine for small models like BGE-small.
  * ``tei_gpu_env``   — CUDA image for Ampere 8.6 GPUs (A10 / A10G).
                        Pick a different tag for your GPU arch:
                          turing-1.9   → T4
                          1.9          → A100 (Ampere 8.0)
                          86-1.9       → A10/A10G (Ampere 8.6)
                          89-1.9       → L4, RTX 40xx (Ada Lovelace)
                          hopper-1.9   → H100

Both set ``command`` (not ``args``) because we are bypassing Flyte's default
``fserve`` entrypoint — the TEI image ships its own ``text-embeddings-router``
binary and knows nothing about Python.

Endpoints exposed by TEI (see https://huggingface.github.io/text-embeddings-inference):
  POST /embed      — returns dense vectors for a list of input strings
  POST /rerank     — cross-encoder scoring for (query, texts) pairs
  POST /predict    — classification head outputs
  GET  /health     — readiness probe
  GET  /info       — model metadata
"""

import flyte
import flyte.app

# Model to serve. Any embedding model on the HF Hub that TEI supports works.
MODEL_ID = "BAAI/bge-small-en-v1.5"

# Flyte apps default to port 8080; bind TEI to the same port so no extra
# plumbing is needed on the platform side.
APP_PORT = 8080


# ---------------------------------------------------------------------------
# CPU variant
# ---------------------------------------------------------------------------

tei_cpu_env = flyte.app.AppEnvironment(
    name="tei-embeddings-cpu",
    image=flyte.Image.from_base("ghcr.io/huggingface/text-embeddings-inference:cpu-1.9"),
    command=f"text-embeddings-router --model-id {MODEL_ID} --port {APP_PORT}",
    port=APP_PORT,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    # Keep one replica warm so the model weights stay in RAM.
    scaling=flyte.app.Scaling(replicas=(1, 2)),
    requires_auth=False,
)


# ---------------------------------------------------------------------------
# GPU variant — A10 / A10G (Ampere 8.6). Swap the tag and resource for other
# architectures (see the module docstring).
# ---------------------------------------------------------------------------

tei_gpu_env = flyte.app.AppEnvironment(
    name="tei-embeddings-gpu",
    image=flyte.Image.from_base("ghcr.io/huggingface/text-embeddings-inference:86-1.9"),
    command=f"text-embeddings-router --model-id {MODEL_ID} --port {APP_PORT}",
    port=APP_PORT,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="A10G:1"),
    scaling=flyte.app.Scaling(replicas=(1, 4)),
    requires_auth=False,
)
