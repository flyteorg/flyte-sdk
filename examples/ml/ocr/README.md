# Batch Document OCR with Qwen Vision-Language Model

Simple, production-ready OCR pipeline using Qwen2.5-VL-3B and Flyte.

## What This Example Shows

- **GPU Workers**: Reusable containers that cache ML models
- **Parallel Processing**: Process multiple documents concurrently
- **DocumentVQA Dataset**: Real document images from HuggingFace
- **Simple Code**: Easy to read and modify

## Quick Start

### 1. Setup

```bash
cd examples/ml/ocr
uv sync
```

### 2. Run OCR on 10 Documents

```bash
flyte run batch_ocr.py run_ocr_pipeline --sample_size=10
```

This will:
1. Download 10 document images from DocumentVQA
2. Extract text using Qwen2.5-VL-3B
3. Return a DataFrame with results

## Architecture

```
┌─────────────────┐
│ Driver          │  Orchestrates workflow
│  - Loads data   │
│  - Splits work  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GPU Workers     │  Process images
│  - Cache model  │  (Model loaded once per worker)
│  - Run OCR      │
└─────────────────┘
```

## Files

- **`batch_ocr.py`**: Main workflow with tasks
- **`ocr_processor.py`**: OCR processing logic
- **`example_usage.py`**: Code examples
- **`README.md`**: This file

## Tasks

### `run_ocr_pipeline(sample_size, chunk_size)`

Complete end-to-end pipeline.

**Inputs:**
- `sample_size`: Number of documents (0 for all)
- `chunk_size`: Documents per batch

**Outputs:**
- DataFrame with OCR results

### `load_dataset(sample_size)`

Download DocumentVQA images.

**Inputs:**
- `sample_size`: Number of images to download

**Outputs:**
- Directory containing images

### `batch_ocr(images_dir, chunk_size)`

Run OCR on a directory of images.

**Inputs:**
- `images_dir`: Directory with images
- `chunk_size`: Documents per batch

**Outputs:**
- DataFrame with OCR results

### `process_batch(image_files)`

Process a batch of images (runs on GPU worker).

**Inputs:**
- `image_files`: List of image files

**Outputs:**
- List of OCR results

## Environments

### `gpu_worker`

GPU worker for Qwen VL OCR processing with model caching.

**Resources:**
- CPU: 4 cores
- Memory: 16Gi
- GPU: 1x T4

**Reuse Policy:**
- 4 parallel workers
- 10 minute idle timeout
- Model cached per worker

### `driver`

Driver for orchestrating batch OCR workflows.

**Resources:**
- CPU: 4 cores
- Memory: 8Gi
- No GPU

## Output Format

Results are returned as a DataFrame:

```python
{
    "document_id": "s3://bucket/doc_001.png",
    "extracted_text": "Full OCR text...",
    "success": True,
    "token_count": 245
}
```

## Usage Examples

### Process Sample Data

```bash
flyte run batch_ocr.py run_ocr_pipeline --sample_size=10 --chunk_size=5
```

### Process Your Own Images

```bash
# Upload local images and run OCR
flyte run batch_ocr.py batch_ocr --images_dir=/path/to/images --chunk_size=20
```

### Programmatic Usage

```python
import asyncio
from pathlib import Path
import flyte
from batch_ocr import run_ocr_pipeline

async def main():
    flyte.init_from_config(root_dir=Path(__file__).parent)

    run = await flyte.run.aio(
        run_ocr_pipeline,
        sample_size=10,
        chunk_size=5,
    )

    await run.wait.aio()
    results = await run.outputs.aio()
    print(results)

asyncio.run(main())
```

## How It Works

1. **Load Dataset**: Downloads images from HuggingFace
2. **Split into Chunks**: Divides images into batches
3. **Parallel Processing**: Sends batches to GPU workers
4. **OCR Extraction**: Each worker runs Qwen VL model
5. **Combine Results**: Merges all results into DataFrame

## Model Caching

The Qwen model is loaded once per GPU worker and cached:

```python
@alru_cache(maxsize=1)
async def get_ocr_processor(model_id):
    return QwenOCRProcessor(model_id)
```

This means:
- First task loads the model (~1 min)
- Subsequent tasks reuse loaded model (~instant)
- Each worker maintains its own cache

## Performance Tips

### Adjust Chunk Size

```bash
# More parallel workers (higher overhead)
--chunk_size=10

# Fewer parallel workers (lower overhead)
--chunk_size=50
```

### Increase Workers

Edit `batch_ocr.py`:

```python
gpu_worker = flyte.TaskEnvironment(
    reusable=flyte.ReusePolicy(
        replicas=8,  # More workers
    ),
)
```

## Troubleshooting

### CUDA Out of Memory

GPU is too small for the model.

**Solution**: Reduce concurrent processing by decreasing `chunk_size`

### Slow Processing

Model running on CPU instead of GPU.

**Check logs for:**
```
Model loaded successfully on cuda  # Good
Model loaded successfully on cpu   # Bad - very slow
```

**Fix**: Ensure Flyte cluster has GPU nodes available

### HuggingFace Token Required

Some models need authentication.

**Setup:**
```bash
# Option 1: Login
huggingface-cli login

# Option 2: Set token
export HF_HUB_TOKEN=your_token
```

## Code Overview

### ocr_processor.py

Simple processor class:

```python
class QwenOCRProcessor:
    def __init__(self, model_id):
        # Load model once

    def extract_text(self, image):
        # Run OCR on single image

    async def process_document(self, image_file):
        # Process single document file
```

### batch_ocr.py

Three simple tasks:

1. `load_dataset`: Download images
2. `process_batch`: Run OCR (GPU worker)
3. `batch_ocr`: Orchestrate batches
4. `run_ocr_pipeline`: End-to-end workflow

## Extending

### Change Model

Edit `batch_ocr.py`:

```python
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # Larger model

gpu_worker = flyte.TaskEnvironment(
    resources=flyte.Resources(
        memory="40Gi",    # More memory
        gpu="A100:1",     # Better GPU
    ),
)
```

### Add Custom Processing

Modify `ocr_processor.py`:

```python
def extract_text(self, image, prompt="Extract all text"):
    # Custom prompt
    # Post-processing
    # Format output
```

## Support

- Issues: https://github.com/flyteorg/flyte/issues
- Docs: https://docs.flyte.org/
- Qwen Model: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
