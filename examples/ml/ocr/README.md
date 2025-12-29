# Batch Document OCR with Multi-Model Comparison

Production-ready batch OCR workflow using Flyte 2.0 with multiple vision-language models, pre-created GPU worker environments, and interactive comparison reports.

## Features

- **8 OCR Models**: Qwen2.5-VL, GOT-OCR 2.0, InternVL 2.5, RolmOCR
- **Pre-Created GPU Workers**: T4, A100, A100 80G (single & multi-GPU)
- **Class-Based Architecture**: Clean separation of OCR logic and workflow orchestration
- **Async Model Caching**: Models loaded once per worker, cached with async LRU
- **DocumentVQA Dataset**: Real document images from HuggingFace (streaming mode)
- **Flexible Scaling**: Sample mode (testing) or full dataset (production)
- **Comparison Reports**: Interactive HTML dashboards with Chart.js

## Quick Start

### 1. Setup

```bash
cd examples/ml/ocr
uv sync --prerelease=allow
```

### 2. Run Single Model (10 documents, ~2 min)

```bash
flyte run batch_ocr.py batch_ocr_single_model \
    --model=QWEN_VL_2B \
    --sample_size=10
```

### 3. Compare Multiple Models (with report)

```bash
flyte run batch_ocr_report.py batch_ocr_comparison_with_report \
    --models='["Qwen/Qwen2.5-VL-2B-Instruct", "OpenGVLab/InternVL2_5-2B"]' \
    --sample_size=20
```

## Supported Models

| Model | GPU | Memory | Best For |
|-------|-----|--------|----------|
| **Qwen2.5-VL 2B** | T4:1 | 16Gi | Lightweight, general docs |
| **Qwen2.5-VL 7B** | A100:1 | 40Gi | Complex layouts, high accuracy |
| **Qwen2.5-VL 72B** | A100 80G:4 | 160Gi | Enterprise-grade, maximum accuracy |
| **GOT-OCR 2.0** | A100:1 | 40Gi | Vision-language grounding |
| **InternVL 2.5 2B** | T4:1 | 16Gi | Fast inference, good quality |
| **InternVL 2.5 8B** | A100:1 | 40Gi | Balanced performance |
| **InternVL 2.5 26B** | A100 80G:2 | 80Gi | High-end document understanding |
| **RolmOCR** | T4:1 | 16Gi | Low-VRAM deployments |

## Architecture

### Two-Layer Design

```
batch_ocr.py                    ocr_processor.py
(Workflow Orchestration)        (OCR Logic)
┌─────────────────────┐         ┌──────────────────────┐
│ Worker Environments │         │  OCRProcessor Class  │
│  • T4              │         │   • __init__()       │
│  • A100            │         │   • run_inference()  │
│  • A100 80G        │◄────────┤   • process_batch()  │
│  • A100 80G Multi  │         │                      │
│                     │         │  @alru_cache         │
│ Dispatcher Tasks    │         │  get_ocr_processor() │
│  • ..._t4()        │         └──────────────────────┘
│  • ..._a100()      │
│  • ..._a100_80g()  │
└─────────────────────┘
```

**Key Design:**
- **Pre-created worker environments**: 4 GPU types, explicit resources
- **Dispatcher tasks**: Route models to appropriate GPU workers
- **OCRProcessor class**: Encapsulates model loading + inference + batch processing
- **Async LRU cache**: Models loaded once, cached by model_id (up to 4 models)

### How It Works

1. **Workflow calls** → `batch_ocr_pipeline(model, images_dir)`
2. **Dispatcher selection** → `get_dispatcher_for_model(model)` → Returns appropriate dispatcher (T4/A100/etc.)
3. **Dispatcher invokes** → `process_document_batch_t4/a100/etc.(model, files)`
4. **Core function gets** → `processor = await get_ocr_processor(model_id)` (cached)
5. **Processor runs** → `processor.process_batch(files, batch_size)` → DataFrame

**Result:** Clean, testable, explicit GPU requirements visible in workflow graph.

## Usage Examples

### Process Full Dataset

```bash
flyte run batch_ocr.py batch_ocr_single_model \
    --model=QWEN_VL_7B \
    --sample_size=0  # 0 = all documents
```

### Process Your Own Images

```bash
flyte run batch_ocr.py batch_ocr_pipeline \
    --model=QWEN_VL_2B \
    --images_dir=<path_to_images> \
    --chunk_size=50
```

### Model Selection Guide

| Use Case | Model | Reason |
|----------|-------|--------|
| Quick testing | QWEN_VL_2B | Fast, T4 GPU |
| Production balanced | QWEN_VL_7B | Good accuracy, A100 |
| Maximum accuracy | QWEN_VL_72B | Best quality, expensive |
| Low VRAM | ROLM_OCR | Optimized for T4 |

## Output

Each run produces a DataFrame with:

```python
{
    "document_id": "s3://bucket/doc_001.png",
    "model": "Qwen/Qwen2.5-VL-2B-Instruct",
    "extracted_text": "Full OCR text...",
    "success": True,
    "error": None,
    "token_count": 245
}
```

## Performance Tips

### Optimal Chunk Size
```bash
--chunk_size=50   # Good for most cases
--chunk_size=20   # More parallelism, higher overhead
--chunk_size=100  # Less parallelism, lower overhead
```

### GPU Utilization

Each worker environment has configurable replicas and concurrency:

```python
worker_env_t4 = flyte.TaskEnvironment(
    reusable=flyte.ReusePolicy(
        replicas=4,      # 4 parallel workers
        concurrency=2,   # 2 tasks per worker
        idle_ttl=600,    # Keep warm for 10 minutes
    )
)
```

**To increase throughput:** Edit replicas (4 → 8) in `batch_ocr.py`

### Batch Size

```bash
# Adjust based on GPU memory
--batch_size=8   # Default for 2B models (T4)
--batch_size=4   # Default for 7B models (A100)
--batch_size=2   # Default for 72B models (A100 80G)
```

## Troubleshooting

### CUDA Out of Memory

**Solution 1:** Reduce batch size
```bash
--batch_size=2  # Instead of default 4
```

**Solution 2:** Use smaller model
```bash
--model=QWEN_VL_2B  # Instead of QWEN_VL_7B
```

### Slow Processing

**Check:** Are GPUs being used?

```bash
# Look for this in logs:
"Model loaded successfully on cuda"  # Good
"Model loaded successfully on cpu"   # Bad - very slow
```

**Fix:** Ensure Flyte cluster has GPU nodes with proper labels

### HuggingFace Authentication

Some models require accepting license:

1. Visit model page: https://huggingface.co/Qwen/Qwen2.5-VL-2B-Instruct
2. Accept license agreement
3. Login: `huggingface-cli login`
4. Or set `HF_TOKEN` environment variable

## Project Structure

```
ocr/
├── ocr_processor.py          # OCR logic (class-based)
│   ├── OCRProcessor          # Main class: model loading, inference, batch processing
│   └── get_ocr_processor()   # Async LRU cache for processor instances
│
├── batch_ocr.py              # Workflow orchestration
│   ├── Worker environments   # 4 pre-created (T4, A100, A100 80G, multi-GPU)
│   ├── Dispatcher tasks      # 4 dispatchers (one per environment)
│   ├── Core function         # 10 lines (was 200+), uses OCRProcessor
│   └── Workflows             # Single model, comparison, pipeline
│
├── batch_ocr_report.py       # Interactive HTML report generation
├── pyproject.toml            # Dependencies (transformers, torch, datasets, etc.)
├── example_usage.py          # Programmatic usage examples
└── README.md                 # This file
```

## Adding New Models

1. **Add to enum:**
```python
class OCRModel(str, Enum):
    MY_MODEL = "my-org/my-model-id"
```

2. **Add GPU config:**
```python
MODEL_GPU_CONFIGS[OCRModel.MY_MODEL] = ModelConfig(
    model_id=OCRModel.MY_MODEL.value,
    gpu_type="A100",
    gpu_count=1,
    memory="40Gi",
    cpu=8,
    max_batch_size=4,
)
```

3. **Done!** Dispatcher auto-selects based on GPU type.

## Adding New GPU Types

1. **Create worker environment:**
```python
worker_env_h100 = flyte.TaskEnvironment(
    name="ocr_worker_h100",
    resources=flyte.Resources(gpu="H100:1", memory="80Gi", cpu=16),
    reusable=flyte.ReusePolicy(replicas=2, concurrency=2),
)
```

2. **Create dispatcher:**
```python
@worker_env_h100.task(cache="auto", retries=3)
async def process_document_batch_h100(...):
    return await _process_document_batch_core(...)
```

3. **Update dispatcher selection:**
```python
def get_dispatcher_for_model(model):
    if config.gpu_type == "H100":
        return process_document_batch_h100
    # ... existing logic ...
```

4. **Update driver dependencies:**
```python
driver_env = flyte.TaskEnvironment(
    depends_on=[..., worker_env_h100]
)
```

## Code Metrics

| Metric | Value |
|--------|-------|
| Core function | 10 lines (was 200+) |
| Worker environments | 4 pre-created |
| Dispatcher tasks | 4 (one per GPU type) |
| Model caching | Async LRU (up to 4 models) |
| Separation of concerns | 2 files: workflow + OCR logic |

## Citations

If you use this workflow, please cite the relevant model papers. See each model's HuggingFace page for citation information.

## License

This example code: Apache 2.0
Individual models: Check each model's HuggingFace page

## Support

- **Issues**: https://github.com/flyteorg/flyte/issues
- **Flyte Docs**: https://docs.flyte.org/
