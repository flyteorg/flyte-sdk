# Image Classification: Fine-tuning, Serving, and Batch Inference

This example demonstrates a complete end-to-end ML workflow in Flyte 2.0:
1. **Training**: Fine-tune a vision transformer model on HuggingFace datasets
2. **Serving**: Deploy the trained model as a FastAPI service
3. **Batch Inference**: Process thousands of images efficiently with reusable containers

## What This Example Shows

### Key Flyte 2.0 Features

1. **Separate Training and Serving Scripts**
   - `training.py`: Contains the model training task
   - `serving.py`: Contains the FastAPI serving application
   - Demonstrates how to structure ML projects with separate concerns

2. **Dependency Isolation**
   - `pyproject.toml` with split optional dependencies
   - Training dependencies (`[training]`): transformers, datasets, accelerate
   - Serving dependencies (`[serving]`): fastapi, uvicorn
   - Minimizes image size and reduces security surface area

3. **RunOutput Integration**
   - Training task outputs a `flyte.io.Dir` containing the fine-tuned model
   - Serving app uses `flyte.app.RunOutput` to reference the training output
   - Automatic model mounting at `/tmp/finetuned_model`

4. **XET Acceleration**
   - Enables fast dataset transfers from HuggingFace
   - Set via environment variable: `HF_XET_HIGH_PERFORMANCE=1`

5. **GPU Resource Management**
   - Training task requests GPU resources
   - Serving app uses CPU only for inference

## Project Structure

```
image_classification/
├── README.md           # This file
├── pyproject.toml      # Dependency management with optional extras
├── training.py         # Model fine-tuning script
├── serving.py          # Model serving API
└── batch_inference.py  # Batch inference with reusable containers
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

### 1. Train the Model

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
- Save the model, processor, and label mappings
- Return a `flyte.io.Dir` containing all artifacts

### 2. Serve the Model

Once training completes, deploy the model as a FastAPI service:

```bash
flyte serve serving.py
```

This will:
- Load the fine-tuned model from the training run output
- Start a FastAPI server with inference endpoints
- Expose the API at a public URL

### 3. Test the API

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

### 4. Batch Inference

For processing large numbers of images efficiently, use the batch inference pipeline:

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
- `model_dir`: Directory with the trained model (from training.py output)
- `images_dir`: Directory containing images to classify
- `chunk_size`: Number of images per parallel task (default: 100)
- `batch_size`: Mini-batch size for GPU processing (default: 32)

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
┌─────────────────────────────────────────┐
│  finetune_image_model Task              │
│                                         │
│  1. Load dataset from HuggingFace       │
│  2. Preprocess images                   │
│  3. Fine-tune ViT model                 │
│  4. Save model + processor + labels     │
│  5. Return flyte.io.Dir                 │
└─────────────────────────────────────────┘
              │
              ▼
     ┌─────────────────┐
     │  Model Artifacts │
     │  (flyte.io.Dir)  │
     └─────────────────┘
```

### Serving Application (`serving.py`)

```
┌─────────────────────────────────────────┐
│  FastAPI App Environment                │
│                                         │
│  Parameters:                            │
│    - model: RunOutput from training     │
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

1. **Separation of Concerns**: Training, serving, and batch inference are independent scripts
2. **Dependency Minimization**: Each script only installs what it needs
3. **Resource Optimization**:
   - Training uses GPU for fine-tuning
   - Serving uses CPU for real-time inference
   - Batch inference uses reusable GPU containers for throughput
4. **Artifact Management**: Models passed between tasks via `flyte.io.Dir`
5. **Environment Configuration**: XET acceleration via environment variables
6. **API Documentation**: OpenAPI/Swagger docs auto-generated
7. **Health Checks**: Proper health endpoints for monitoring
8. **Reusable Containers**: Model loaded once, amortized across many images
9. **Efficient Batching**: Chunk-level and mini-batch-level batching for GPU utilization
10. **Comprehensive Reporting**: Detailed statistics and error tracking

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

### Model Not Loading in Serving

Check that:
1. Training completed successfully
2. The `task_name` in `serving.py` matches your training task
3. The model directory is mounted correctly

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
