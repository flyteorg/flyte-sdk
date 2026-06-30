# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.5.0",
#     "torchvision",
#     "fastapi",
#     "uvicorn",
#     "pillow",
#     "python-multipart",
#     "flyte==2.0.9",
# ]
# ///
"""
Custom Model Serving with torch.export (.pt2 format)

This example shows how to serve a PyTorch model that you load yourself —
no AutoModel, no from_pretrained, no assumptions about model format.

The pattern directly mirrors how Ray Serve works:
- Ray Serve: model is loaded in a Ray Actor's __init__, stays resident in GPU memory
- Union:      model is loaded in @env.on_startup, stays resident in GPU memory

The key concepts demonstrated:
  1. torch.export (.pt2) format: a serialized computation graph, exported on CPU for portability
  2. move_to_device_pass: rewrites the graph to run on GPU — necessary because device
     placement is baked into .pt2 graph nodes, so .to("cuda") alone will NOT work
  3. Custom transforms: manual preprocessing instead of AutoImageProcessor
  4. Warmup forward pass: compiles CUDA kernels at startup, not on the first real request
  5. Always-on replicas: Scaling(replicas=(1, N)) keeps instances warm with no cold start

Requires PyTorch >= 2.5 (torch.export.passes.move_to_device_pass was added in 2.5).

Usage:
    # Export the model — run once, result is stored in Union's object store
    flyte run custom_model_serving.py export_model_to_pt2

    # Deploy the serving app
    flyte serve custom_model_serving.py env
"""

import io
import logging
import pathlib

import torch
import torchvision.models as models
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

import flyte
import flyte.app
import flyte.io
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared image — built from this script's inline dependencies above.
# Used by both the export task and the serving app.
# ---------------------------------------------------------------------------

image = flyte.Image.from_uv_script(__file__, name="cv-model-server")

# ---------------------------------------------------------------------------
# Export task
#
# Exports a ResNet-50 to .pt2 format on CPU. We export on CPU so the artifact
# is portable — the serving app uses move_to_device_pass to retarget to GPU
# at runtime without needing to re-export.
#
# ---------------------------------------------------------------------------

task_env = flyte.TaskEnvironment(
    name="cv-export",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)


@task_env.task
def export_model_to_pt2() -> flyte.io.File:
    """
    Export a ResNet-50 to .pt2 (torch.export) format on CPU.

    torch.export serializes the full computation graph, including all device
    placement. Exporting on CPU is the recommended portable pattern: load on
    any machine, then use move_to_device_pass to retarget to the serving GPU.

    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    model.cpu()

    # Provide a concrete example input matching the expected shape and dtype.
    example_input = torch.randn(1, 3, 224, 224)
    exported = torch.export.export(model, (example_input,))

    output_path = "./model.pt2"
    torch.export.save(exported, output_path)
    logger.info("Exported model to %s", output_path)

    return flyte.io.File.from_local_sync(output_path)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

fastapi_app = FastAPI(
    title="Custom Model Server",
    description="Serve a .pt2 model with a custom loader",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# App environment
#
# Declares resources, scaling behavior, and how the model artifact gets into
# the container. Equivalent to a Ray Serve deployment config.
# ---------------------------------------------------------------------------

env = FastAPIAppEnvironment(
    name="custom-model-server",
    app=fastapi_app,
    description="Serve a .pt2 CV model with a custom loader",
    image=image,
    # Choose a GPU.
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="A10G:1"),
    requires_auth=False,
    # Always-on: keep at least 1 warm replica so there is no cold start.
    # Upper bound of 4 lets the app scale out under load.
    scaling=flyte.app.Scaling(replicas=(1, 4)),
    parameters=[
        flyte.app.Parameter(
            # 'model' matches the argument name in @env.on_startup below.
            # Union downloads the file and injects it as a flyte.io.File.
            name="model",
            value=flyte.app.RunOutput(
                task_name="cv-export.export_model_to_pt2",
                type="file",
            ),
            download=True,
        )
    ],
)


# ---------------------------------------------------------------------------
# Startup hook
#
# Here, @env.on_startup runs once before uvicorn
# starts, loads the model into GPU memory, and stores it on fastapi_app.state
# so every request can access it directly — no loading latency per request.
#
# The `model` argument is injected by Union: it matches the Parameter named
# "model" defined above, downloaded to a local path and wrapped as flyte.io.File.
# ---------------------------------------------------------------------------


@env.on_startup
async def startup(model: flyte.io.File):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model on device: %s", device)

    # Step 1: Load the ExportedProgram from the .pt2 file.

    exported = torch.export.load(model.path)
    logger.info("Loaded ExportedProgram from %s", model.path)

    # Step 2: Retarget the graph to the serving device.

    if device.type != "cpu":
        from torch.export.passes import move_to_device_pass

        exported = move_to_device_pass(exported, device)
        logger.info("Applied move_to_device_pass -> %s", device)

    # Step 3: Extract a callable nn.Module from the ExportedProgram.

    fastapi_app.state.model = exported.module()

    # Step 4: Define preprocessing transforms.

    fastapi_app.state.transforms = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    fastapi_app.state.device = device

    # Step 5: Warmup forward pass.

    logger.info("Running warmup forward pass...")
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        fastapi_app.state.model(dummy)
    logger.info("Model ready.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@fastapi_app.get("/health")
async def health():
    """
    Indicates whether the model has finished loading and which device it's on.
    Poll this after deployment to confirm the instance is warm before sending
    real traffic.
    """
    model_loaded = getattr(fastapi_app.state, "model", None) is not None
    return {
        "status": "ready" if model_loaded else "starting",
        "model_loaded": model_loaded,
        "device": str(getattr(fastapi_app.state, "device", "unknown")),
    }


@fastapi_app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Run inference on an uploaded image.

    Accepts a JPEG or PNG image and returns the top-5 predicted class indices
    with confidence scores. Replace the postprocessing with your own logic.
    """
    if getattr(fastapi_app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    tensor = fastapi_app.state.transforms(pil_image).unsqueeze(0).to(fastapi_app.state.device)

    with torch.no_grad():
        logits = fastapi_app.state.model(tensor)

    probs = torch.softmax(logits, dim=-1)[0]
    top5 = probs.topk(5)

    return {
        "predictions": [
            {"class_index": int(idx), "confidence": round(float(score), 4)}
            for idx, score in zip(top5.indices.tolist(), top5.values.tolist())
        ]
    }


# ---------------------------------------------------------------------------
# Main — export then deploy
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
    )

    # Step 1: Export the model — run once, result is stored in Union's object store.
    run = flyte.run(export_model_to_pt2)
    print(f"Export run: {run.url}")
    run.wait()

    # Step 2: Deploy the serving app.
    deployed = flyte.deploy(env)
    print(f"App deployed: {deployed[0].table_repr()}")
