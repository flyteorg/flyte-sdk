# Image Classification: Artifacts, Triggers, and Lineage

This example demonstrates a complete end-to-end ML workflow in Flyte 2.0, wired
together by **artifacts** rather than by hard-coded run names:

1. **Training**: Fine-tune a vision transformer and publish it as a *model artifact*
2. **Batch Inference**: Auto-triggered by every new model version; publishes its
   predictions as a *data artifact*
3. **Serving**: Deploy a FastAPI app that binds to the model artifact by name

### The artifact graph

```
       flyte run training.py finetune_image_model
                        │
                        ▼
        ┌───────────────────────────────┐
        │  artifact: beans-classifier   │  kind=model, HTML model card,
        │  (a flyte.io.Dir of weights)  │  attrs: base_model, dataset, accuracy…
        └───────────────────────────────┘
              │                     │
   OnArtifact trigger fires         │  flyte.app.ArtifactValue
              ▼                     ▼
  ┌───────────────────────┐   ┌──────────────────────────┐
  │ batch_inference_demo  │   │ image-classification-api │
  │ (scores held-out set) │   │ (FastAPI, /predict)      │
  └───────────────────────┘   └──────────────────────────┘
              │
              ▼
  ┌──────────────────────────────────────┐
  │ artifact: beans-classifier-predictions│  kind=data
  │ (a flyte.io.DataFrame)                │
  └──────────────────────────────────────┘
```

Nothing downstream names a run, a run output, or even the training task. Retrain,
and batch inference re-scores automatically; redeploy the app, and it picks up the
latest weights. Every hop is recorded, so you can walk a prediction set back to the
exact model version that produced it, and that model version back to the action
and inputs that trained it.

## What This Example Shows

### Key Flyte 2.0 Features

1. **Artifacts as the integration surface**
   - `finetune_image_model` is marked `produces_artifacts=True` and returns
     `artifacts.new(model_dir, metadata)` — the platform registers a new version of
     `beans-classifier` and records the producing action as its source
   - `Metadata.create_model_metadata(...)` stamps `kind="model"` plus searchable
     attrs (base model, dataset, epochs, measured accuracy)
   - An HTML model card, rendered from the *real* evaluation metrics, is attached
     and shown in the UI (HTML rather than markdown: HTML cards render in an
     iframe, markdown cards need CORS configured on the bucket)

2. **Artifact triggers (`OnArtifact`)**
   - `batch_inference_demo` carries a `flyte.Trigger` whose automation is
     `flyte.OnArtifact(name="beans-classifier")` — any new version fires a run
   - `flyte.TriggeredArtifact` marks which input the fresh artifact binds to
     (`model_dir`); the rest of the trigger's `inputs` are ordinary defaults
   - Triggers are registered by `flyte deploy`, not by `flyte run`

3. **Lineage**
   - Batch inference consumes the model artifact and publishes
     `beans-classifier-predictions`, so the platform links model → predictions
   - The serving app's `ArtifactValue` records the resolved artifact version,
     giving app → model lineage
   - `flyte get artifact --source-run <run>` walks the graph from the other end

4. **Separate Training, Inference, and Serving Scripts**
   - `training.py`, `batch_inference.py`, `serving.py`: separate concerns, separate
     images, separate dependency extras
   - `model_artifact.py` holds only the shared artifact *names* — the three scripts
     never import each other

5. **Dependency Isolation**
   - `pyproject.toml` with split optional dependencies
   - Training dependencies (`[training]`): transformers, datasets, accelerate
   - Serving dependencies (`[serving]`): fastapi, uvicorn
   - Minimizes image size and reduces security surface area

6. **XET Acceleration**
   - Enables fast dataset transfers from HuggingFace
   - Set via environment variable: `HF_XET_HIGH_PERFORMANCE=1`

7. **GPU Resource Management**
   - Training and batch inference request GPU resources
   - Serving app uses CPU only for inference

## Project Structure

```
image_classification/
├── README.md           # This file
├── pyproject.toml      # Dependency management with optional extras
├── model_artifact.py   # Shared artifact names (the only cross-script coupling)
├── training.py         # Fine-tuning; publishes the model artifact
├── serving.py          # Model serving API; binds to the artifact by name
└── batch_inference.py  # Artifact-triggered batch inference + reusable containers
```

## Installation

### For Training Only
```bash
uv pip install .[training]
```

### For Serving Only
```bash
uv pip install .[serving]
```

### For Batch Inference Only
```bash
uv pip install .[batch]
```

### For All Components (Local Development)
```bash
uv pip install .[all]
```

## Usage

### 1. Register the trigger (once)

Deploy the batch-inference environment so the `score-on-new-model` trigger is
registered with the platform. Triggers live on *deployed* tasks — `flyte run`
alone never registers one:

```bash
flyte deploy batch_inference.py driver_with_report_env
```

Confirm it landed:

```bash
flyte get trigger
```

### 2. Train the Model

Run the training script to fine-tune a ViT-tiny model on the beans dataset:

```bash
flyte run training.py finetune_image_model
```

**With custom parameters:**
```bash
flyte run training.py finetune_image_model \
    --dataset_name="food101" \
    --num_epochs=5 \
    --batch_size=32 \
    --learning_rate=1e-4
```

**Available datasets:** `beans`, `cifar10`, `food101`, `imagenet-1k`, etc.

The training task will:
- Download and preprocess the dataset with XET acceleration
- Fine-tune a small ViT model (WinKawaks/vit-tiny-patch16-224)
- Evaluate on the validation split and render an HTML model card from the results
- Save the model, processor, and label mappings
- Publish it all as a new version of the `beans-classifier` model artifact

Inspect what was published:

```bash
flyte get artifact beans-classifier          # every version, newest first
flyte get artifact beans-classifier <ver>    # one version's details + card
flyte get artifact --kind model              # all model artifacts
```

**Watch the trigger fire.** Shortly after the training action succeeds, a run of
`batch_inference_demo` appears that nobody launched — the platform bound the fresh
artifact to its `model_dir` input. That run publishes
`beans-classifier-predictions`, whose lineage points back at the model version it
scored with.

> **If no run appears:** the training env sets `cache=flyte.Cache("auto", "1.0")`.
> Re-running with identical inputs is a cache hit, which reuses the previous
> outputs and so publishes **no new artifact version** — and a trigger fires on new
> versions only. Vary an input (`--num_epochs=4`) to get a genuinely new model.

### 3. Serve the Model

Deploy the model as a FastAPI service:

```bash
flyte serve serving.py env
```

This will:
- Resolve `beans-classifier` from the artifact service **at deploy time** and mount
  it at `/tmp/finetuned_model`
- Start a FastAPI server with inference endpoints
- Expose the API at a public URL

`ArtifactValue(version=None)` resolves the latest version and *pins* it, so a later
training run cannot swap the weights under a live app. Redeploy to move forward, or
pass `version="<artifact-version>"` to serve a specific checkpoint.

### 4. Test the API

**Using the interactive docs:**
```
Open: <app-url>/docs
```

**Using curl:**
```bash
curl -X POST <app-url>/predict \
    -F "file=@/path/to/image.jpg"
```

**Using Python:**
```python
import requests

url = "<app-url>/predict"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### 5. Batch Inference (on demand)

Step 1 registered the trigger, so batch inference normally runs *by itself*. To
kick one off by hand — for a different dataset, or an older checkpoint — bind the
artifact directly:

```bash
python batch_inference.py
```

which is just:

```python
from flyte.remote import Artifact

model = Artifact.get("beans-classifier")            # or version="<artifact-version>"
flyte.run(batch_inference_demo, model_dir=model)    # binds to the `Dir` input
```

An `Artifact` binds straight to a typed task input, and the version consumed is
recorded on the run — which is what makes the lineage query work later. That beats
the alternative of listing successful runs of a named training task and fishing the
`Dir` out of `run.outputs()[0]`: no dependence on the producing task's name, and no
guessing about which output slot held the model.

For processing large numbers of images from a directory you already have, the
lower-level pipeline still takes a plain `Dir`:

```bash
flyte run batch_inference.py batch_inference_pipeline \
    --model_dir=<model_directory> \
    --images_dir=<images_directory> \
    --chunk_size=100 \
    --batch_size=32
```

**What it does:**
- Discovers all images in the directory (supports: .jpg, .jpeg, .png, .bmp, .gif, .tiff)
- Partitions images into chunks for parallel processing
- Processes each chunk using reusable GPU containers
- Generates a comprehensive report with all predictions

**Key parameters:**
- `model_dir`: Directory with the trained model (an `Artifact`, or a raw `Dir`)
- `images_dir`: Directory containing images to classify
- `chunk_size`: Number of images per parallel task (default: 100)
- `batch_size`: Mini-batch size for GPU processing (default: 32)

### 6. Follow the lineage

```bash
# Which prediction sets exist, and what produced them?
flyte get artifact beans-classifier-predictions

# Everything a given training run published
flyte get artifact --source-run <training-run-name>

# All models trained on a particular dataset
flyte get artifact --kind model --attr dataset=AI-Lab-Makerere/beans
```

**Performance:**
- **Reusable containers**: Model loaded once, reused for all chunks
- **8 replicas × 2 concurrency = 16 parallel tasks**
- **GPU utilization**: Batch processing maximizes GPU efficiency
- **Example**: 10,000 images processed in ~15 minutes on 8 GPUs

**Output Report:**
```json
{
  "summary": {
    "total_images": 10000,
    "successful": 9987,
    "failed": 13,
    "success_rate": 0.9987
  },
  "confidence_stats": {
    "average": 0.92,
    "min": 0.45,
    "max": 0.99
  },
  "label_distribution": {
    "bean_rust": 4521,
    "healthy": 3210,
    "angular_leaf_spot": 2256
  },
  "predictions": [...]
}
```

## API Endpoints

### `GET /health`
Health check endpoint returning model status and available classes.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 3,
  "classes": ["angular_leaf_spot", "bean_rust", "healthy"]
}
```

### `POST /predict`
Classify an uploaded image.

**Request:** Multipart form data with image file

**Response:**
```json
{
  "predictions": [
    {"label": "bean_rust", "confidence": 0.95},
    {"label": "angular_leaf_spot", "confidence": 0.03},
    {"label": "healthy", "confidence": 0.02}
  ],
  "top_prediction": {
    "label": "bean_rust",
    "confidence": 0.95
  }
}
```

### `GET /classes`
Get all available classification classes.

**Response:**
```json
{
  "num_classes": 3,
  "classes": ["angular_leaf_spot", "bean_rust", "healthy"]
}
```

## Architecture

### Training Pipeline (`training.py`)

```
┌──────────────────────────────────────────────────┐
│  finetune_image_model (produces_artifacts=True)  │
│                                                  │
│  1. Load dataset from HuggingFace                │
│  2. Preprocess images                            │
│  3. Fine-tune ViT model                          │
│  4. Evaluate → metrics for the card + attrs      │
│  5. Save model + processor + labels              │
│  6. Upload HTML model card                       │
│  7. Return artifacts.new(Dir, Metadata)          │
└──────────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │  artifact: beans-classifier@<v>  │
   │  kind=model · card · attrs       │
   │  source = this action            │
   └──────────────────────────────────┘
```

### Serving Application (`serving.py`)

```
┌─────────────────────────────────────────┐
│  FastAPI App Environment                │
│                                         │
│  Parameters:                            │
│    - model: ArtifactValue(              │
│        "beans-classifier", directory)   │
│      resolved + pinned at deploy time   │
│                                         │
│  Mounted at: /tmp/finetuned_model       │
└─────────────────────────────────────────┘
              │
              ▼
     ┌──────────────────┐
     │  Model Loading    │
     │  (lifespan hook)  │
     └──────────────────┘
              │
              ▼
     ┌──────────────────┐
     │  Inference API    │
     │  /predict         │
     │  /health          │
     │  /classes         │
     └──────────────────┘
```

### Batch Inference Pipeline (`batch_inference.py`)

```
   ┌──────────────────────────────────┐
   │  artifact: beans-classifier@<v>  │
   └──────────────────────────────────┘
              │  OnArtifact trigger — binds the new version
              │  to `model_dir` via flyte.TriggeredArtifact
              ▼
┌─────────────────────────────────────────────────┐
│  batch_inference_demo (triggered)               │
│    → extract_dataset_images                     │
│    → batch_inference_with_report                │
└─────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  Driver Task (batch_inference_pipeline)         │
│                                                 │
│  1. Discover all images in directory            │
│  2. Partition into chunks (e.g., 100 per chunk) │
│  3. Launch parallel worker tasks                │
│  4. Aggregate results into report               │
└─────────────────────────────────────────────────┘
              │
              ▼
     ┌────────────────────────────────────┐
     │  Chunk 0   Chunk 1   ...  Chunk N  │  (Parallel)
     └────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────┐
│  Worker Tasks (process_image_batch)              │
│                                                  │
│  Reusable Container Configuration:               │
│    - 8 replicas                                  │
│    - 2 concurrency per replica                   │
│    - Model loaded once (lru_cache)               │
│    - GPU batch processing                        │
│                                                  │
│  Process:                                        │
│    1. Load model (cached)                        │
│    2. Process images in mini-batches             │
│    3. Return predictions as JSON                 │
└──────────────────────────────────────────────────┘
              │
              ▼
     ┌──────────────────────┐
     │  Prediction Files     │
     │  (flyte.io.File)      │
     └──────────────────────┘
              │
              ▼
     ┌──────────────────────┐
     │  Final Report         │
     │  - Summary stats      │
     │  - Label distribution │
     │  - All predictions    │
     └──────────────────────┘
              │
              ▼
  ┌────────────────────────────────────────┐
  │ artifact: beans-classifier-predictions │  kind=data
  │ (flyte.io.DataFrame)                   │  lineage → the model version
  └────────────────────────────────────────┘  it was scored with
```

**Reusable Container Benefits:**
- **Model loading amortized**: Loaded once per container, reused for all chunks
- **High throughput**: 16 concurrent tasks (8 replicas × 2 concurrency)
- **GPU efficiency**: Batch processing within each chunk
- **Cost-effective**: Containers stay alive for multiple tasks

## Configuration

### Training Configuration

- **Model**: `WinKawaks/vit-tiny-patch16-224` (22M parameters)
- **Resources**: 4 CPU, 16Gi memory, 1 GPU
- **Caching**: Enabled with auto versioning
- **Environment**: `HF_XET_HIGH_PERFORMANCE=1`

### Serving Configuration

- **Model**: Loaded from training run output
- **Resources**: 2 CPU, 4Gi memory (CPU-only inference)
- **Authentication**: Disabled (set `requires_auth=True` for production)

### Batch Inference Configuration

**Worker Environment:**
- **Resources**: 2 CPU, 8Gi memory, 1 GPU
- **Reusable Policy**:
  - 8 replicas (parallel workers)
  - 2 concurrency per replica
  - 300s idle TTL (keep containers warm)
  - 300s scaledown TTL

**Driver Environment:**
- **Resources**: 2 CPU, 4Gi memory (orchestration only)
- **Depends on**: Worker environment (ensures workers are ready)

**Processing Configuration:**
- **Chunk size**: 100 images per task (configurable)
- **Batch size**: 32 images per GPU batch (configurable)
- **Retry policy**: 3 retries per chunk

## Best Practices Demonstrated

1. **Couple on artifact names, not run names**: nothing downstream references a run
   ID, a run output slot, or the producing task's name — rename or rewrite
   `training.py` and the trigger and the app keep working
2. **Publish metadata with the weights**: a model card rendered from the *measured*
   evaluation, plus searchable attrs, so `flyte get artifact --kind model` is a
   usable model registry rather than a list of opaque directories
3. **Automate the fan-out**: `OnArtifact` means evaluation happens on every new
   model without a human remembering to launch it
4. **Publish downstream results as artifacts too**: predictions are a first-class
   asset, and their lineage answers "which weights produced this?" months later
5. **Pin what serves traffic**: `ArtifactValue` resolves *and pins* at deploy time,
   so a training run can never silently swap the weights under a live app
6. **Separation of Concerns**: Training, serving, and batch inference are independent scripts
7. **Dependency Minimization**: Each script only installs what it needs
8. **Resource Optimization**:
   - Training uses GPU for fine-tuning
   - Serving uses CPU for real-time inference
   - Batch inference uses reusable GPU containers for throughput
9. **Environment Configuration**: XET acceleration via environment variables
10. **API Documentation**: OpenAPI/Swagger docs auto-generated
11. **Health Checks**: Proper health endpoints for monitoring
12. **Reusable Containers**: Model loaded once, amortized across many images
13. **Efficient Batching**: Chunk-level and mini-batch-level batching for GPU utilization
14. **Comprehensive Reporting**: Detailed statistics and error tracking

## Customization

### Using Different Models

Replace the model name in `training.py`:
```python
model_name = "google/vit-base-patch16-224"  # Larger model
# or
model_name = "microsoft/resnet-50"          # ResNet architecture
```

### Using Different Datasets

Change the dataset in the training command:
```bash
flyte run training.py finetune_image_model --dataset_name="cifar10"
```

### Adjusting Resources

Modify the resource requests in either file:
```python
resources=flyte.Resources(cpu=8, memory="32Gi", gpu=2)
```

## Troubleshooting

### Out of Memory During Training

Reduce batch size:
```bash
flyte run training.py finetune_image_model --batch_size=16
```

### The Trigger Never Fires

Work through these in order:

1. **Was the trigger registered?** `flyte get trigger` should list
   `score-on-new-model`. If not, run
   `flyte deploy batch_inference.py driver_with_report_env`. A `flyte run` does not
   register triggers.
2. **Was a new artifact version actually published?**
   `flyte get artifact beans-classifier` — if the version count did not grow, the
   training run was a cache hit (see step 2 of Usage) or the task is missing
   `produces_artifacts=True`.
3. **Do the names match?** The `OnArtifact(name=...)` in `batch_inference.py` and the
   `Metadata(name=...)` in `training.py` both read `MODEL_ARTIFACT_NAME` from
   `model_artifact.py` — a divergence here is silent, the trigger just never fires.
4. **Same project and domain?** Artifact names are scoped to project/domain, so a
   model published in `development` will not fire a trigger deployed in `production`.

### Model Not Loading in Serving

Check that:
1. Training completed successfully **and published an artifact** —
   `flyte get artifact beans-classifier` should list at least one version
2. The artifact name in `serving.py` (`MODEL_ARTIFACT_NAME`) matches what training
   published, in the same project/domain
3. The model directory is mounted correctly at `/tmp/finetuned_model`

A deploy that fails with `ParameterMaterializationError` means the artifact could
not be resolved at all — usually nothing has been published under that name yet, so
train first.

### Slow Dataset Loading

Ensure XET is enabled:
```bash
export HF_XET_HIGH_PERFORMANCE=1
```

### Batch Inference Running Slowly

**Increase parallelism:**
```python
# In batch_inference.py, modify worker_env:
reusable=flyte.ReusePolicy(
    replicas=16,      # More replicas
    concurrency=4,    # Higher concurrency
)
```

**Increase chunk size:**
```bash
flyte run batch_inference.py batch_inference_pipeline \
    --chunk_size=200  # Larger chunks, fewer tasks
```

### Out of Memory in Batch Inference

**Reduce batch size:**
```bash
flyte run batch_inference.py batch_inference_pipeline \
    --batch_size=16  # Smaller GPU batches
```

**Or reduce chunk size:**
```bash
flyte run batch_inference.py batch_inference_pipeline \
    --chunk_size=50  # Smaller chunks
```

## Learn More

- [Flyte 2.0 Documentation](https://docs.flyte.org)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

### Related examples

- `examples/artifacts/artifact_example.py` — every way an artifact can be born
  (File, Dir, DataFrame, multi-output tasks, explicit `Artifact.create`)
- `examples/triggers/artifact_trigger.py` — the smallest possible `OnArtifact` demo
- `examples/triggers/artifact_trigger_external.py` — triggers fired by publishes
  that come from outside Flyte entirely (CLI, or an external system's callback)
