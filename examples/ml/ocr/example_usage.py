"""
Example Usage Script for Batch OCR Workflows

This script demonstrates how to run the OCR workflows programmatically.
"""

from pathlib import Path

import flyte
import flyte.remote

from batch_ocr import (
    OCRModel,
    batch_ocr_comparison,
    batch_ocr_single_model,
    load_documentvqa_dataset,
)
from batch_ocr_report import batch_ocr_comparison_with_report


async def example_single_model():
    """Run OCR with a single model on sample data."""
    print("Example 1: Single Model OCR")
    print("=" * 50)

    # Run with Qwen 2B on 10 documents
    result = await batch_ocr_single_model(
        model=OCRModel.QWEN_VL_3B,
        sample_size=10,
        chunk_size=5,
    )

    print(f"Completed! Result type: {type(result)}")
    print("Check the Flyte UI for the DataFrame output")


async def example_multi_model_comparison():
    """Compare multiple models on the same dataset."""
    print("\nExample 2: Multi-Model Comparison")
    print("=" * 50)

    # Compare two lightweight models
    models = [OCRModel.QWEN_VL_3B, OCRModel.INTERN_VL_2B]

    results = await batch_ocr_comparison(
        models=models,
        sample_size=20,
        chunk_size=10,
    )

    print(f"Completed! Got {len(results)} result DataFrames (one per model)")
    print("Check the Flyte UI for detailed results")


async def example_with_report():
    """Run comparison with interactive report."""
    print("\nExample 3: Comparison with Interactive Report")
    print("=" * 50)

    # This will generate a beautiful HTML report
    results = await batch_ocr_comparison_with_report(
        models=["Qwen/Qwen2.5-VL-3B-Instruct", "OpenGVLab/InternVL2_5-2B"],
        sample_size=15,
        chunk_size=5,
    )

    print(f"Completed with report! Got {len(results)} result DataFrames")
    print("Check the Flyte UI for the interactive HTML report")


async def example_custom_dataset():
    """Load dataset separately and then run OCR."""
    print("\nExample 4: Custom Dataset Loading")
    print("=" * 50)

    # Load dataset
    print("Loading DocumentVQA dataset...")
    images_dir = await load_documentvqa_dataset(sample_size=5, split="train")

    print(f"Dataset loaded to: {images_dir}")
    print("You can now run batch_ocr_pipeline with this directory")


def example_via_flyte_run():
    """Show how to trigger workflows via flyte.run()"""
    print("\nExample 5: Using flyte.run() API")
    print("=" * 50)

    # Initialize Flyte
    flyte.init_from_config(root_dir=Path(__file__).parent)

    # Run single model workflow
    run = flyte.run(
        batch_ocr_single_model,
        model=OCRModel.QWEN_VL_3B,
        sample_size=5,
        chunk_size=5,
    )

    print(f"Workflow submitted: {run.url}")
    print("Monitor progress in the Flyte UI")

    # You can also wait for results
    # outputs = run.outputs()
    # print(f"Results: {outputs}")


def example_model_configs():
    """Show how to inspect model configurations."""
    print("\nExample 6: Inspecting Model Configurations")
    print("=" * 50)

    from batch_ocr import MODEL_GPU_CONFIGS

    print("Available Models and Their GPU Requirements:\n")

    for model, config in MODEL_GPU_CONFIGS.items():
        print(f"{model.name}:")
        print(f"  Model ID: {config.model_id}")
        print(f"  GPU: {config.gpu_type} x{config.gpu_count}")
        print(f"  Memory: {config.memory}")
        print(f"  CPU: {config.cpu}")
        print(f"  Max Batch Size: {config.max_batch_size}")
        print(f"  Flash Attention: {config.requires_flash_attention}")
        print()


if __name__ == "__main__":
    print("Batch OCR Workflow Examples")
    print("=" * 50)
    print()

    # Example 6 is synchronous - show model configs
    example_model_configs()

    print("\n" + "=" * 50)
    print("To run the async examples, use one of these methods:")
    print("=" * 50)
    print()
    print("1. Via Flyte CLI (recommended):")
    print("   flyte run batch_ocr.py batch_ocr_single_model --model=QWEN_VL_3B --sample_size=10")
    print()
    print("2. Via flyte.run() API:")
    print("   python example_usage.py  # (uncomment example_via_flyte_run() in main)")
    print()
    print("3. Direct async execution (requires Flyte environment):")
    print("   asyncio.run(example_single_model())")
    print()
    print("=" * 50)

    # Uncomment to run via flyte.run():
    # example_via_flyte_run()

    # Uncomment to run directly (requires proper Flyte setup):
    # asyncio.run(example_single_model())
    # asyncio.run(example_multi_model_comparison())
    # asyncio.run(example_with_report())
