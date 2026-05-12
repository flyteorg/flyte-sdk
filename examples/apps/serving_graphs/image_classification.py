"""
Serving graph — CPU pre/post split from a GPU forward pass.

This example shows the canonical "two-app" inference graph: heavy CPU work
on one app, the GPU forward pass on another, talking to each other over HTTP
inside the cluster.

Why split? In a typical vision/audio/feature-engineering pipeline the GPU
forward pass is fast (millis) but is sandwiched between slow CPU work
(image decode, resize, denoise, NMS, label lookup, etc.). If you put both
stages in one process you pay for an idle GPU during preprocessing. Splitting
them lets each side scale independently:

    client ──► [cpu_app  x N replicas]  ──► [gpu_app x M replicas] ──► back
                preprocess + postprocess        model.forward only
                cheap CPU, scale wide           expensive GPU, scale narrow

Wire format between the two apps is raw float32 bytes (not JSON) — for
anything tensor-shaped this is the single biggest perf knob.
"""

import io
import logging
import pathlib
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image, ImageFilter
from pydantic import BaseModel

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
# Shared base with the deps both apps need (HTTP server + numpy). The CPU and
# GPU images extend it with their own disjoint stacks — the CPU app never
# imports torch and the GPU app never imports PIL. Sharing the base layer
# means the registry only stores one copy of fastapi/uvicorn/numpy.

base_image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "fastapi",
    "uvicorn",
    "numpy",
)

cpu_image = base_image.with_pip_packages(
    "httpx",
    "pillow",
)

gpu_image = base_image.with_pip_packages(
    "torch==2.7.1",
    "torchvision==0.22.1",
)

# ---------------------------------------------------------------------------
# Shared tensor layout
# ---------------------------------------------------------------------------

INPUT_C, INPUT_H, INPUT_W = 3, 224, 224
NUM_CLASSES = 1000
TENSOR_DTYPE = np.float32


# ===========================================================================
# GPU app — model.forward only
# ===========================================================================


@asynccontextmanager
async def _gpu_lifespan(app: FastAPI):
    # Imported lazily so the CPU app never has to import torch.
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.to("cuda")
    app.state.model = model
    app.state.device = device
    app.state.categories = list(weights.meta["categories"])
    logging.getLogger(__name__).info("model loaded on %s", device)
    yield


gpu_app = FastAPI(
    title="inference-gpu",
    description="ResNet18 forward pass.",
    lifespan=_gpu_lifespan,
)


@gpu_app.get("/health")
async def gpu_health() -> dict:
    return {"status": "ok", "device": gpu_app.state.device}


@gpu_app.get("/labels")
async def labels() -> list[str]:
    # Exposed so the CPU side can fetch labels once at startup instead of
    # hard-coding the ImageNet class list.
    return gpu_app.state.categories


@gpu_app.post("/infer")
async def infer(request: Request) -> Response:
    """Run a batched forward pass.

    Request body:  raw float32 bytes, shape (B, 3, 224, 224), C-contiguous.
    Response body: raw float32 bytes, shape (B, 1000) — raw logits.

    We deliberately do NOT use JSON here. For a batch of 32 images the tensor
    is ~19MB; JSON-serializing that is the dominant cost end-to-end.
    """
    import torch

    raw = await request.body()
    arr = np.frombuffer(raw, dtype=TENSOR_DTYPE)
    if arr.size % (INPUT_C * INPUT_H * INPUT_W) != 0:
        raise HTTPException(400, "payload size is not a multiple of one image tensor")
    batch = arr.reshape(-1, INPUT_C, INPUT_H, INPUT_W)

    x = torch.from_numpy(batch).to(gpu_app.state.device)
    with torch.inference_mode():
        logits = gpu_app.state.model(x)
    out = logits.detach().to("cpu").numpy().astype(TENSOR_DTYPE, copy=False)
    return Response(content=out.tobytes(), media_type="application/octet-stream")


gpu_env = FastAPIAppEnvironment(
    name="serving-graph-gpu",
    app=gpu_app,
    image=gpu_image,
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu="A10G:1"),
    # GPU replicas are expensive; keep at least one warm so model weights stay
    # resident, and cap the max. Bump if a single replica saturates.
    scaling=flyte.app.Scaling(replicas=(1, 2)),
    requires_auth=True,
)


# ===========================================================================
# CPU app — pre/postprocess, calls the GPU app
# ===========================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=TENSOR_DTYPE).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=TENSOR_DTYPE).reshape(3, 1, 1)


class ClassifyRequest(BaseModel):
    image_url: str
    top_k: int = 5


class Prediction(BaseModel):
    label: str
    score: float


def _preprocess(img_bytes: bytes) -> np.ndarray:
    """Decode → denoise → resize → normalize. CPU-bound, deliberately so.

    Real preprocessing stacks (detection, OCR, audio) do substantially more
    than this — sliding window crops, color-space conversion, etc. The point
    is that none of it benefits from a GPU sitting next to it.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    img = img.resize((INPUT_W, INPUT_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=TENSOR_DTYPE) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr, dtype=TENSOR_DTYPE)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@asynccontextmanager
async def _cpu_lifespan(app: FastAPI):
    # Resolved at serving time via the cluster-internal endpoint pattern,
    # so this stays correct across local/remote deploys without an env var.
    gpu_url = gpu_env.endpoint
    log = logging.getLogger(__name__)
    log.info("resolved GPU endpoint: %s", gpu_url)
    async with httpx.AsyncClient(timeout=30.0) as bootstrap:
        try:
            r = await bootstrap.get(f"{gpu_url}/labels")
            r.raise_for_status()
        except (httpx.HTTPError, OSError) as e:
            # Most common reason on a fresh deploy: GPU replica hasn't finished
            # pulling its image / loading weights yet. Crash-looping is fine —
            # the next attempt will likely succeed — but make the cause obvious.
            log.error("downstream GPU app at %s not ready: %s", gpu_url, e)
            raise
        app.state.labels = r.json()
    # One persistent client per replica — avoids TCP/TLS handshake per request,
    # which matters once you're doing 100s of req/s.
    async with httpx.AsyncClient(
        base_url=gpu_url,
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
    ) as client:
        app.state.client = client
        yield


cpu_app = FastAPI(
    title="inference-cpu",
    description="Pre/post around the GPU forward pass.",
    lifespan=_cpu_lifespan,
)


@cpu_app.get("/health")
async def cpu_health() -> dict:
    return {"status": "ok", "labels_loaded": len(cpu_app.state.labels)}


@cpu_app.post("/classify", response_model=list[Prediction])
async def classify(req: ClassifyRequest) -> list[Prediction]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        img_resp = await client.get(req.image_url)
        img_resp.raise_for_status()

    tensor = _preprocess(img_resp.content)  # heavy CPU
    batch = tensor[np.newaxis, ...]  # add batch dim

    gpu_resp = await cpu_app.state.client.post(
        "/infer",
        content=batch.tobytes(),
        headers={"content-type": "application/octet-stream"},
    )
    gpu_resp.raise_for_status()
    logits = np.frombuffer(gpu_resp.content, dtype=TENSOR_DTYPE).reshape(1, NUM_CLASSES)

    probs = _softmax(logits, axis=-1)[0]  # back to CPU work
    top_idx = np.argsort(-probs)[: req.top_k]
    return [Prediction(label=cpu_app.state.labels[i], score=float(probs[i])) for i in top_idx]


cpu_env = FastAPIAppEnvironment(
    name="serving-graph-cpu",
    app=cpu_app,
    image=cpu_image,
    resources=flyte.Resources(cpu=4, memory="4Gi"),
    # Cheap, so scale wide. Use scale-to-zero (replicas=(0, 8)) for bursty
    # traffic; keep replicas=(1, 8) here to avoid cold starts in the demo.
    scaling=flyte.app.Scaling(replicas=(1, 8)),
    requires_auth=True,
    depends_on=[gpu_env],
)


# ===========================================================================
# Deploy
# ===========================================================================

if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.INFO,
    )
    app = flyte.serve(cpu_env)
    print(f"Deployed serving graph; public CPU endpoint: {app.url}")
    print("Try: curl -X POST $URL/classify -H 'content-type: application/json' \\")
    print(
        '       -d \'{"image_url": "https://upload.wikimedia.org/wikipedia/commons/4/41/Sunflower_from_Silesia2.jpg"}\''
    )
