# Quick Start Guide - Batch OCR with Flyte 2.0

This guide will get you running batch document OCR with multiple models in minutes.

## Setup

### 1. Install Dependencies

```bash
# From the examples/ml/ocr directory
uv sync --prerelease=allow

# Or with specific extras
uv sync --extra ocr --prerelease=allow
```

### 2. Verify Installation

```bash
# Check that imports work
python3 batch_ocr.py

# Should display usage instructions
```

## Running Workflows

### Example 1: Single Model on Sample Data (Fastest)

Process 10 documents with the lightweight Qwen 2B model:

```bash
flyte run batch_ocr.py batch_ocr_single_model \
    --model=QWEN_VL_2B \
    --sample_size=10 \
    --chunk_size=5
```

**Expected Output:**
- Downloads DocumentVQA dataset (streaming)
- Processes 10 documents in 2 chunks (5 docs each)
- Returns DataFrame with OCR results
- Total time: ~2-5 minutes (depending on GPU availability)

### Example 2: Compare Two Models

Compare Qwen 2B vs InternVL 2B on 20 documents:

```bash
flyte run batch_ocr.py batch_ocr_comparison \
    --models='["QWEN_VL_2B", "INTERN_VL_2B"]' \
    --sample_size=20 \
    --chunk_size=10
```

**Expected Output:**
- Runs both models in parallel on the same documents
- Returns 2 DataFrames (one per model)
- Useful for quick model benchmarking

### Example 3: Full Workflow with Interactive Report

Run comparison and generate beautiful HTML report:

```bash
flyte run batch_ocr_report.py batch_ocr_comparison_with_report \
    --models='["Qwen/Qwen2.5-VL-2B-Instruct", "OpenGVLab/InternVL2_5-2B"]' \
    --sample_size=15 \
    --chunk_size=5
```

**Expected Output:**
- Comparison matrix with statistics
- Interactive charts (success rates, token counts)
- Side-by-side text comparisons
- Error analysis per model

### Example 4: Process Your Own Images

If you have a directory of document images:

```bash
# First, upload your images to Flyte storage or use local path
flyte run batch_ocr.py batch_ocr_pipeline \
    --model=QWEN_VL_2B \
    --images_dir=<path_to_images> \
    --chunk_size=50 \
    --batch_size=8
```

### Example 5: Production Scale

Process the full DocumentVQA dataset with a high-accuracy model:

```bash
flyte run batch_ocr.py batch_ocr_single_model \
    --model=QWEN_VL_7B \
    --sample_size=0 \  # 0 = process all documents
    --chunk_size=100
```

**Note:** This will process thousands of documents. Ensure you have sufficient GPU resources.

## Model Selection Guide

| Use Case | Recommended Model | GPU | Memory | Speed |
|----------|------------------|-----|--------|-------|
| **Quick Testing** | QWEN_VL_2B | T4 | 16Gi | Fast |
| **Balanced Performance** | QWEN_VL_7B | A100 | 40Gi | Medium |
| **Maximum Accuracy** | QWEN_VL_72B | 4x A100 80G | 160Gi | Slow |
| **Lightweight Deployment** | ROLM_OCR | T4 | 16Gi | Fast |
| **Vision-Language Grounding** | GOT_OCR_2 | A100 | 40Gi | Medium |
| **General Purpose** | INTERN_VL_8B | A100 | 40Gi | Medium |

## Understanding the Output

### DataFrame Schema

Each OCR result includes:

```python
{
    "document_id": "s3://bucket/doc_00001.png",  # Document path
    "model": "Qwen/Qwen2.5-VL-2B-Instruct",      # Model used
    "extracted_text": "Full OCR text...",         # Extracted text
    "success": True,                              # Success flag
    "error": None,                                # Error message (if failed)
    "token_count": 245                            # Tokens generated
}
```

### Viewing Results

After running a workflow, you'll get a URL like:

```
https://flyte.example.com/console/projects/flytesnacks/domains/development/executions/f123...
```

Click the URL to see:
- **Execution graph** showing task progress
- **Output DataFrames** with all OCR results
- **Interactive reports** (if using report workflows)
- **Logs** for debugging

## Performance Tips

### 1. Optimal Chunk Size

```bash
# Too small = overhead
--chunk_size=10

# Too large = less parallelism
--chunk_size=500

# Sweet spot (usually)
--chunk_size=50-100
```

### 2. GPU Utilization

```bash
# Increase replicas for more parallelism
# Edit batch_ocr.py:
reusable=flyte.ReusePolicy(
    replicas=8,  # Increase from 4 to 8
    concurrency=2,
)
```

### 3. Batch Size

```bash
# Larger batch = better GPU utilization but more memory
--batch_size=8  # Default for 2B models
--batch_size=4  # Default for 7B models
--batch_size=2  # Default for 72B models
```

## Troubleshooting

### Error: CUDA Out of Memory

**Solution 1:** Use a smaller batch size
```bash
# Edit the workflow call or model config
--batch_size=2
```

**Solution 2:** Use a smaller model
```bash
--model=QWEN_VL_2B  # Instead of QWEN_VL_7B
```

**Solution 3:** Override with more GPU memory
```bash
--overrides='{"process_document_batch": {"resources": {"gpu": "A100 80G:1"}}}'
```

### Error: Module Not Found

**Solution:** Reinstall dependencies
```bash
uv sync --prerelease=allow --extra ocr
```

### Slow Processing

**Cause:** Running on CPU instead of GPU

**Check logs for:**
```
Model loaded on CPU  # Bad - very slow
Model loaded on GPU  # Good - fast
```

**Solution:** Ensure Flyte cluster has GPU nodes and task specifies GPU resources.

### HuggingFace Authentication Required

Some models (like Qwen) require accepting the license:

1. Visit: https://huggingface.co/Qwen/Qwen2.5-VL-2B-Instruct
2. Accept the license agreement
3. Login: `huggingface-cli login`
4. Or set `HF_TOKEN` environment variable

## Next Steps

### Customize Models

Add your own model to `batch_ocr.py`:

```python
class OCRModel(str, Enum):
    MY_MODEL = "my-org/my-ocr-model"

MODEL_GPU_CONFIGS[OCRModel.MY_MODEL] = ModelConfig(
    model_id=OCRModel.MY_MODEL.value,
    gpu_type="A100",
    gpu_count=1,
    memory="40Gi",
    cpu=8,
    max_batch_size=4,
)
```

### Integrate with Your Pipeline

```python
from batch_ocr import batch_ocr_pipeline, OCRModel

@my_env.task
async def my_workflow():
    # Run OCR as part of larger pipeline
    ocr_results = await batch_ocr_pipeline(
        model=OCRModel.QWEN_VL_7B,
        images_dir=my_images,
        chunk_size=100,
    )

    # Process results further
    analyzed = await analyze_ocr_results(ocr_results)
    return analyzed
```

### Generate Custom Reports

```python
from batch_ocr_report import generate_ocr_comparison_report

@my_env.task(report=True)
async def custom_comparison():
    results = [df1, df2, df3]  # Your OCR results
    await generate_ocr_comparison_report(results)
```

## Additional Resources

- **Full Documentation:** See [README.md](README.md)
- **Model Details:** Check each model's HuggingFace page for capabilities
- **Flyte Docs:** https://docs.flyte.org/
- **Report Examples:** Run with `--help` flag to see example commands

## Example Session

Complete end-to-end example:

```bash
# 1. Setup
cd examples/ml/ocr
uv sync --prerelease=allow

# 2. Quick test (2 min)
flyte run batch_ocr.py batch_ocr_single_model \
    --model=QWEN_VL_2B \
    --sample_size=5

# 3. Check results in Flyte UI
# Click the URL in the output

# 4. Run comparison with report (5 min)
flyte run batch_ocr_report.py batch_ocr_comparison_with_report \
    --models='["Qwen/Qwen2.5-VL-2B-Instruct", "OpenGVLab/InternVL2_5-2B"]' \
    --sample_size=20

# 5. View interactive report in browser
# Check the "Reports" tab in Flyte UI

# 6. Iterate and optimize
# Adjust chunk_size, batch_size, models as needed
```

That's it! You're ready to process documents at scale with Flyte 2.0. 🚀
