"""
Example Usage for Batch OCR

Simple examples showing how to run the OCR pipeline.
"""

import asyncio
from pathlib import Path

import flyte

from batch_ocr import run_ocr_pipeline, batch_ocr, load_dataset


async def example_small_batch():
    """Process a small batch of 10 documents."""
    print("Processing 10 documents from DocumentVQA")

    flyte.init_from_config(root_dir=Path(__file__).parent)

    run = await flyte.run.aio(
        run_ocr_pipeline,
        sample_size=10,
        chunk_size=5,
    )

    print(f"Run URL: {run.url}")
    await run.wait.aio()

    results = await run.outputs.aio()
    print(f"Complete! Results: {results}")


async def example_custom_images():
    """Process custom images from a local directory."""
    print("Processing custom images")

    flyte.init_from_config(root_dir=Path(__file__).parent)

    # Upload your local images
    images_dir = await flyte.io.Dir.from_local(Path("./my_images"))

    # Run OCR
    run = await flyte.run.aio(batch_ocr, images_dir=images_dir, chunk_size=10)

    print(f"Run URL: {run.url}")
    await run.wait.aio()

    results = await run.outputs.aio()
    print(f"Complete! Results: {results}")


if __name__ == "__main__":
    print("""
Batch OCR Examples
==================

Run these examples to see the OCR pipeline in action.
""")

    # Run the small batch example
    asyncio.run(example_small_batch())

    # To process custom images:
    # asyncio.run(example_custom_images())
