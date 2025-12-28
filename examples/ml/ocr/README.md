# Batch Document OCR with Multi-Model Comparison

This example demonstrates a production-ready batch document OCR workflow using Flyte 2.0, featuring:

- **Multiple OCR Models**: Qwen2.5-VL, GOT-OCR 2.0, InternVL 2.5, RolmOCR, and more
- **Reusable Containers**: GPU models loaded once and reused across batches
- **Intelligent Caching**: Content-based caching for efficiency
- **DocumentVQA Dataset**: Real-world document images from HuggingFace
- **Flexible Scaling**: Sample mode for testing, full dataset for production
- **GPU Optimization**: Per-model GPU configurations with dynamic overrides
- **Comparison Reports**: Interactive HTML dashboards comparing model performance

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Driver (Orchestration)                                      │
│  ├─ Load DocumentVQA Dataset (streaming)                   │
│  ├─ Partition into chunks                                  │
│  └─ Coordinate parallel processing                         │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┬────────────────┐
        ▼                 ▼                 ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Worker 1     │  │ Worker 2     │  │ Worker 3     │  │ Worker 4     │
│ GPU: A100    │  │ GPU: A100    │  │ GPU: T4      │  │ GPU: T4      │
│ Model: Qwen  │  │ Model: Qwen  │  │ Model: Intern│  │ Model: ROLM  │
│ Cache: ✓     │  │ Cache: ✓     │  │ Cache: ✓     │  │ Cache: ✓     │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │                │
        └─────────────────┴─────────────────┴────────────────┘
                          ▼
        ┌─────────────────────────────────────────┐
        │ Aggregation & Report Generation         │
        │  ├─ Combine results from all workers    │
        │  ├─ Calculate comparison metrics        │
        │  └─ Generate interactive HTML report    │
        └─────────────────────────────────────────┘
```

## Supported Models

| Model | License | GPU Requirement | Memory | Best For |
|-------|---------|----------------|---------|----------|
| **Qwen2.5-VL 2B** | Apache-2.0/Qwen | T4 x1 | 16Gi | Lightweight OCR, general docs |
| **Qwen2.5-VL 7B** | Apache-2.0/Qwen | A100 x1 | 40Gi | Complex layouts, high accuracy |
| **Qwen2.5-VL 72B** | Apache-2.0/Qwen | A100 80G x4 | 160Gi | Enterprise-grade, max accuracy |
| **GOT-OCR 2.0** | MIT | A100 x1 | 40Gi | Vision-language grounding |
| **InternVL 2.5 2B** | MIT | T4 x1 | 16Gi | Fast inference, good quality |
| **InternVL 2.5 8B** | MIT | A100 x1 | 40Gi | Balanced performance |
| **InternVL 2.5 26B** | MIT | A100 80G x2 | 80Gi | High-end document understanding |
| **RolmOCR** | Apache-2.0 | T4 x1 | 16Gi | Low-VRAM deployments |

## Quick Start

### 1. Single Model OCR on Sample Data

Process 10 documents with Qwen2.5-VL 2B:

```bash
flyte run batch_ocr.py batch_ocr_single_model \
    --model=QWEN_VL_2B \
    --sample_size=10
```

### 2. Multi-Model Comparison

Compare two models on 50 documents:

```bash
flyte run batch_ocr.py batch_ocr_comparison \
    --models='["QWEN_VL_2B", "INTERN_VL_2B"]' \
    --sample_size=50
```

This generates an interactive comparison report showing:
- Success rate per model
- Average tokens extracted
- Side-by-side text comparisons
- Error analysis

### 3. Full Dataset Processing

Process entire DocumentVQA dataset:

```bash
flyte run batch_ocr.py batch_ocr_single_model \
    --model=QWEN_VL_7B \
    --sample_size=0  # 0 means process all
```

### 4. Custom Image Directory

Run OCR on your own images:

```bash
flyte run batch_ocr.py batch_ocr_pipeline \
    --model=QWEN_VL_2B \
    --images_dir=<flyte_dir_or_local_path> \
    --chunk_size=50
```

### 5. Generate Comparison Report

```bash
flyte run batch_ocr_report.py batch_ocr_comparison_with_report \
    --models='["Qwen/Qwen2.5-VL-2B-Instruct", "OpenGVLab/InternVL2_5-2B"]' \
    --sample_size=50
```

## Advanced Usage

### Dynamic GPU Override

Override GPU resources at runtime for specific workloads:

```bash
flyte run batch_ocr.py batch_ocr_single_model \
    --model=QWEN_VL_7B \
    --sample_size=100 \
    --overrides='{"process_document_batch": {"resources": {"gpu": "A100 80G:2", "memory": "80Gi"}}}'
```

### Sample vs Full Dataset Toggle

The `sample_size` parameter controls dataset size:

- `sample_size=10`: Process 10 documents (testing)
- `sample_size=100`: Process 100 documents (benchmarking)
- `sample_size=0`: Process entire dataset (production)

### Switching Models

Models are defined as an Enum in `batch_ocr.py`. To add a new model:

```python
class OCRModel(str, Enum):
    MY_NEW_MODEL = "huggingface/model-id"

# Add GPU config
MODEL_GPU_CONFIGS[OCRModel.MY_NEW_MODEL] = ModelConfig(
    model_id=OCRModel.MY_NEW_MODEL.value,
    gpu_type="A100",
    gpu_count=1,
    memory="40Gi",
    cpu=8,
    max_batch_size=4,
)
```

### Adjusting Parallelism

Control chunk size and batch size for different workloads:

```bash
# High parallelism (many small chunks)
flyte run batch_ocr.py batch_ocr_single_model \
    --model=QWEN_VL_2B \
    --sample_size=1000 \
    --chunk_size=50  # 20 parallel chunks

# Lower parallelism (fewer large chunks)
flyte run batch_ocr.py batch_ocr_single_model \
    --model=QWEN_VL_2B \
    --sample_size=1000 \
    --chunk_size=200  # 5 parallel chunks
```

## Key Features

### 1. Reusable Containers

Models are loaded once per worker replica and cached with `@lru_cache`:

```python
@lru_cache(maxsize=1)
def load_ocr_model(model_id: str):
    # Model loaded once, reused across all tasks in this replica
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model
```

Worker configuration:

```python
reusable=flyte.ReusePolicy(
    replicas=4,       # 4 parallel workers
    concurrency=2,    # Each handles 2 tasks concurrently
    idle_ttl=600,     # Keep alive 10 minutes after idle
)
```

### 2. Content-Based Caching

Tasks are cached based on inputs:

```python
@default_worker_env.task(cache="auto", retries=3)
async def process_document_batch(...):
    # Results cached - rerun with same inputs = instant results
```

### 3. Streaming Dataset Loading

DocumentVQA dataset loaded in streaming mode for memory efficiency:

```python
dataset = load_dataset("HuggingFaceM4/DocumentVQA", split="train", streaming=True)
```

### 4. Error Handling

Individual document failures don't stop the pipeline:

```python
# Track success/failure per document
{
    "document_id": "doc_001.png",
    "success": False,
    "error": "CUDA out of memory",
    "extracted_text": ""
}
```

### 5. Interactive Reports

Generate beautiful HTML reports with Chart.js:

- **Comparison Matrix**: Success rates, token counts, totals
- **Charts**: Success rates, token distributions
- **Side-by-Side**: Text comparisons across models
- **Error Analysis**: Breakdown of failures by type

## Output Format

### DataFrame Schema

Each OCR run produces a DataFrame with:

| Column | Type | Description |
|--------|------|-------------|
| `document_id` | string | Document identifier/path |
| `model` | string | Model used for OCR |
| `extracted_text` | string | OCR output text |
| `success` | bool | Whether OCR succeeded |
| `error` | string | Error message (if failed) |
| `token_count` | int | Number of tokens generated |

### Example Output

```python
{
    "document_id": "s3://my-bucket/docs/invoice_001.png",
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "extracted_text": "INVOICE\nDate: 2024-01-15\nAmount: $1,234.56...",
    "success": True,
    "error": None,
    "token_count": 245
}
```

## Performance Optimization Tips

### 1. Choose the Right Model

- **T4 models** (2B variants): Best for high throughput, simple documents
- **A100 models** (7B-8B variants): Balanced performance and accuracy
- **Multi-GPU models** (26B-72B variants): Maximum accuracy, complex layouts

### 2. Tune Chunk Size

- **Small chunks (20-50)**: Better parallelism, higher overhead
- **Large chunks (100-200)**: Lower overhead, less parallelism
- **Sweet spot**: Usually 50-100 documents per chunk

### 3. Adjust Reusable Container Replicas

```python
reusable=flyte.ReusePolicy(
    replicas=8,  # Increase for more parallelism (if you have GPUs)
    concurrency=2,
)
```

### 4. Enable Flash Attention

For supported models (Qwen, InternVL large variants):

```bash
pip install flash-attn --no-build-isolation
```

Set `requires_flash_attention=True` in `ModelConfig`.

## Troubleshooting

### Out of Memory (OOM)

**Problem**: Task fails with CUDA OOM error

**Solution**:
1. Reduce `batch_size` in the task call
2. Use a smaller model (e.g., 2B instead of 7B)
3. Override with more GPU memory:
   ```bash
   --overrides='{"process_document_batch": {"resources": {"gpu": "A100 80G:1"}}}'
   ```

### Slow Processing

**Problem**: OCR is too slow

**Solution**:
1. Increase `chunk_size` to reduce overhead
2. Increase `replicas` in `ReusePolicy`
3. Use a lighter model (2B variants)
4. Ensure GPU acceleration is working (check logs for "Model loaded on GPU")

### Model Download Errors

**Problem**: HuggingFace model download fails

**Solution**:
1. Check HuggingFace authentication: `huggingface-cli login`
2. For gated models (Qwen), accept license on HuggingFace website
3. Add `HF_TOKEN` environment variable to Flyte task

### No GPUs Available

**Problem**: Running on CPU (very slow)

**Solution**:
1. Ensure Flyte cluster has GPU nodes
2. Check `resources.gpu` specification in task
3. Verify GPU drivers on cluster nodes

## Project Structure

```
ocr/
├── batch_ocr.py              # Main workflow (models, tasks, pipelines)
├── batch_ocr_report.py       # Report generation
├── pyproject.toml            # Dependencies
├── README.md                 # This file
└── examples/                 # Example outputs (if any)
```

## Citation

If you use this workflow, please cite the relevant model papers:

**Qwen2.5-VL**:
```
@article{qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Qwen Team},
  year={2024}
}
```

**GOT-OCR 2.0**:
```
@article{got-ocr2,
  title={General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model},
  author={Wang, Haoran and others},
  year={2024}
}
```

**InternVL 2.5**:
```
@article{internvl2_5,
  title={InternVL 2.5: Empowering Multimodal Large Language Models},
  author={Chen, Zhe and others},
  year={2024}
}
```

## License

This example code is provided under the Apache 2.0 license. Individual models have their own licenses - please check each model's HuggingFace page for details.

## Contributing

Contributions are welcome! To add a new OCR model:

1. Add the model to the `OCRModel` enum
2. Add GPU configuration to `MODEL_GPU_CONFIGS`
3. Test with sample data
4. Update this README with model details

## Support

For issues or questions:
- Flyte SDK: https://github.com/flyteorg/flyte
- This example: [Create an issue](https://github.com/flyteorg/flyte/issues)
