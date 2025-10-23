# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "flyte>=2.0.0b25",
#    "transformers>=4.46.3",
#    "torch>=2.6.0",
#    "Pillow",
#    "pdf2image",
#    "aiohttp",
#    "einops",
# ]
# ///

"""
Flyte-Native DeepSeek-OCR Pipeline for arXiv Papers

This Flyte pipeline efficiently converts arXiv papers (PDFs) into text using:
- DeepSeek-OCR (3B param model) with Flash Attention 2
- Torch compilation for optimized inference
- Flyte's reusable tasks with GPU acceleration
- AsyncIO for concurrent processing
- Batch processing for optimal GPU utilization
- Efficient memory management and caching

Model: deepseek-ai/DeepSeek-OCR (3B params, MIT license)
Paper: https://arxiv.org/abs/2510.18234

Requirements:
- transformers>=4.46.3
- torch>=2.6.0
- Pillow, pdf2image, aiohttp, einops
- flash-attn>=2.7.3 (GPU container only)
- addict, easydict (GPU container only)
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import aiohttp
import torch
from PIL import Image
from pdf2image import convert_from_bytes
from transformers import AutoProcessor, AutoModelForVision2Seq

import flyte
import flyte.io

# Configure logging
logger = logging.getLogger(__name__)

# Create image from script dependencies
# Add flash-attn and other GPU-specific packages to container image
image = flyte.Image.from_uv_script(__file__, name="deepseek_ocr_image").with_pip_packages(
    "unionai-reuse>=0.1.5",
    "flash-attn>=2.7.3",  # Only in container, not local, optimizes the model for lower memory usage and speed
    "addict",
    "easydict",
)

# Driver environment for orchestration (CPU-only)
driver = flyte.TaskEnvironment(
    name="arxiv_ocr_driver",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

# Worker environment for OCR with GPU
# DeepSeek-OCR (3B) works well on L4 (24GB) with Flash Attention
# L4 is more cost-effective than A10g while providing good performance
ocr_worker = flyte.TaskEnvironment(
    name="arxiv_ocr_worker",
    image=image,
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
        gpu="L4:1",  # L4 (24GB) - optimal for DeepSeek-OCR 3B with Flash Attention
    ),
    reusable=flyte.ReusePolicy(
        replicas=8,           # Scale up to 8 parallel OCR tasks
        concurrency=1,        # One paper per replica
        idle_ttl=300,         # Keep warm for 5 minutes
        scaledown_ttl=300
    ),
)


@dataclass
class ArxivPaper:
    """Represents an arXiv paper to be processed."""
    arxiv_id: str
    title: str
    pdf_url: str


@dataclass
class OCRResult:
    """Result of OCR processing for a paper."""
    arxiv_id: str
    title: str
    content: str
    num_pages: int
    processing_time: float
    success: bool
    error: Optional[str] = None


@lru_cache(maxsize=1)
def get_ocr_model(model_name: str = "deepseek-ai/DeepSeek-OCR"):
    """
    Lazily load and cache the OCR model with torch compilation for fast inference.

    DeepSeek-OCR: 3B param model with Flash Attention 2
    Paper: https://arxiv.org/abs/2510.18234

    The model is compiled using torch.compile for optimized inference:
    - First inference will be slower (compilation time)
    - Subsequent inferences are significantly faster
    - Uses bfloat16 for optimal GPU memory usage
    """
    logger.info(f"Loading OCR model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Try to use Flash Attention 2 if available (in container)
    # Fall back to eager attention for local development
    attn_impl = "flash_attention_2"
    try:
        import flash_attn
        logger.info("Flash Attention 2 detected, using optimized attention")
    except ImportError:
        logger.warning("Flash Attention 2 not available, using eager attention (slower)")
        attn_impl = "eager"

    # Load model with bfloat16
    logger.info(f"Loading model with {attn_impl} attention...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
        attn_implementation=attn_impl
    )
    model.eval()

    # Compile model for optimized inference (only on CUDA)
    if device == "cuda":
        logger.info("Compiling model with torch.compile (mode='reduce-overhead')...")
        logger.info("Note: First inference will be slow due to compilation")

        try:
            # Use 'reduce-overhead' mode for best throughput
            # Alternative: 'max-autotune' for maximum speed (longer compile time)
            model = torch.compile(
                model,
                mode="reduce-overhead",  # Balance between compile time and runtime
                fullgraph=True,          # Try to compile entire graph
                dynamic=False            # Static shapes for better optimization
            )
            logger.info("Model compiled successfully!")
        except Exception as e:
            logger.warning(f"Torch compilation failed, falling back to eager mode: {e}")
    else:
        logger.info("CPU detected, skipping torch.compile (GPU-only optimization)")

    logger.info(f"Model loaded on device: {device}")
    return processor, model, device


async def download_pdf(pdf_url: str) -> bytes:
    """
    Download PDF content from URL.

    Args:
        pdf_url: URL to PDF file

    Returns:
        PDF content as bytes
    """
    async with aiohttp.ClientSession() as session:
        logger.info(f"Downloading PDF from {pdf_url}")
        async with session.get(pdf_url, timeout=aiohttp.ClientTimeout(total=120)) as response:
            response.raise_for_status()
            pdf_bytes = await response.read()
            logger.info(f"Downloaded {len(pdf_bytes) / 1024 / 1024:.2f} MB")
            return pdf_bytes


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    """
    Convert PDF bytes to images (one per page).

    Args:
        pdf_bytes: PDF content
        dpi: Resolution for conversion

    Returns:
        List of PIL Images
    """
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    logger.info(f"Converted PDF to {len(images)} images")
    return images


def ocr_images_batch(
    images: List[Image.Image],
    model_name: str,
    batch_size: int = 2  # DeepSeek-OCR works best with batch_size=1 or 2
) -> List[str]:
    """
    Perform OCR on a batch of images using DeepSeek-OCR.

    Args:
        images: List of PIL Images
        model_name: HuggingFace model identifier (default: deepseek-ai/DeepSeek-OCR)
        batch_size: Number of pages to process in one batch (default: 2)

    Returns:
        List of extracted text strings

    Note:
        - First batch will be slower due to torch.compile
        - Subsequent batches will be significantly faster
        - Uses bfloat16 and Flash Attention 2 for efficiency
    """
    processor, model, device = get_ocr_model(model_name)
    texts = []

    # Warmup: Run a dummy forward pass to trigger compilation
    # This makes the first real inference faster
    if len(images) > 0:
        logger.info("Running warmup inference to trigger torch.compile...")
        try:
            dummy_input = processor(
                images=[images[0]],
                return_tensors="pt"
            ).pixel_values.to(device)

            with torch.no_grad():
                _ = model.generate(
                    pixel_values=dummy_input,
                    max_new_tokens=10,  # Short warmup
                    do_sample=False
                )
            logger.info("Warmup complete! Model is now compiled and optimized.")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    # Process in batches for efficiency
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} ({len(batch_images)} pages)")

        # Prepare inputs
        inputs = processor(
            images=batch_images,
            return_tensors="pt"
        )

        # Move all inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Generate text with DeepSeek-OCR optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,     # Longer sequences for documents
                num_beams=1,             # Greedy decoding for speed
                do_sample=False,         # Deterministic output
                use_cache=True,          # Enable KV caching
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # Decode outputs
        batch_texts = processor.batch_decode(outputs, skip_special_tokens=True)
        texts.extend(batch_texts)

        # Clear cache to prevent OOM
        if device == "cuda":
            torch.cuda.empty_cache()

    return texts


@ocr_worker.task(cache="auto", retries=3)
async def process_single_paper(
    arxiv_id: str,
    title: str,
    pdf_url: str,
    model_name: str = "deepseek-ai/DeepSeek-OCR",
    batch_size: int = 2,  # DeepSeek-OCR works best with smaller batches
    dpi: int = 200
) -> flyte.io.File:
    """
    Flyte task to process a single arXiv paper through the complete OCR pipeline.

    Args:
        arxiv_id: arXiv paper ID
        title: Paper title
        pdf_url: URL to PDF
        model_name: HuggingFace model for OCR
        batch_size: Number of pages to process in one batch
        dpi: Resolution for PDF to image conversion

    Returns:
        flyte.io.File: File containing the OCR result as JSON
    """
    start_time = time.time()

    try:
        # Step 1: Download PDF
        pdf_bytes = await download_pdf(pdf_url)

        # Step 2: Convert to images (CPU-bound, run in executor)
        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(None, pdf_to_images, pdf_bytes, dpi)
        num_pages = len(images)
        logger.info(f"Converted {arxiv_id} to {num_pages} images")

        # Step 3: Perform OCR
        logger.info(f"Starting OCR for {arxiv_id}")
        page_texts = await loop.run_in_executor(
            None,
            ocr_images_batch,
            images,
            model_name,
            batch_size
        )

        # Combine all pages
        full_text = "\n\n".join([
            f"--- Page {i+1} ---\n{text}"
            for i, text in enumerate(page_texts)
        ])

        processing_time = time.time() - start_time
        logger.info(f"✓ Completed {arxiv_id} in {processing_time:.2f}s")

        result = OCRResult(
            arxiv_id=arxiv_id,
            title=title,
            content=full_text,
            num_pages=num_pages,
            processing_time=processing_time,
            success=True
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"✗ Failed {arxiv_id}: {e}")
        result = OCRResult(
            arxiv_id=arxiv_id,
            title=title,
            content="",
            num_pages=0,
            processing_time=processing_time,
            success=False,
            error=str(e)
        )

    # Save result to file
    out_path = os.path.join(tempfile.gettempdir(), f"{arxiv_id}_ocr.json")
    with open(out_path, 'w') as f:
        json.dump(result.__dict__, f, indent=2)

    return await flyte.io.File.from_local(out_path)


@driver.task(cache="auto")
async def process_arxiv_papers(
    papers: List[ArxivPaper],
    model_name: str = "deepseek-ai/DeepSeek-OCR",
    batch_size: int = 2,
    dpi: int = 200
) -> List[flyte.io.File]:
    """
    Driver task to orchestrate OCR processing of multiple arXiv papers.

    This task distributes papers across GPU workers for parallel processing using
    DeepSeek-OCR (3B param model) with torch compilation and Flash Attention 2.

    Args:
        papers: List of ArxivPaper objects to process
        model_name: HuggingFace model for OCR (default: DeepSeek-OCR 3B)
        batch_size: Number of pages to process in one batch (default: 2 for DeepSeek-OCR)
        dpi: Resolution for PDF conversion

    Returns:
        List of flyte.io.File containing OCR results

    Performance:
        - First inference per worker: ~2-3s (includes compilation)
        - Subsequent inferences: ~0.5-1s per page
        - Scales linearly with number of GPU replicas (up to 8)
    """
    logger.info(f"Processing {len(papers)} arXiv papers with model {model_name}")

    # Create tasks for all papers using async for loop pattern
    tasks = []
    for paper in papers:
        with flyte.group(f"paper-{paper.arxiv_id}"):
            task = asyncio.create_task(
                process_single_paper(
                    paper.arxiv_id,
                    paper.title,
                    paper.pdf_url,
                    model_name,
                    batch_size,
                    dpi
                )
            )
            tasks.append(task)

    # Process papers concurrently and collect results as they complete
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        logger.info(f"Completed paper, {len(results)}/{len(papers)} done")

    logger.info(f"All {len(papers)} papers processed!")
    return results


async def main():
    """Main entry point for the OCR pipeline."""

    # Sample arXiv papers
    papers = [
        ArxivPaper(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf"
        ),
        ArxivPaper(
            arxiv_id="1810.04805",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            pdf_url="https://arxiv.org/pdf/1810.04805.pdf"
        ),
        ArxivPaper(
            arxiv_id="2005.14165",
            title="GPT-3: Language Models are Few-Shot Learners",
            pdf_url="https://arxiv.org/pdf/2005.14165.pdf"
        ),
    ]

    # Initialize Flyte
    await flyte.init_from_config.aio()

    # Run the pipeline
    print(f"\n{'='*60}")
    print(f"Processing {len(papers)} arXiv papers...")
    print(f"Model: DeepSeek-OCR (3B params)")
    print(f"Features: Flash Attention 2 + torch.compile")
    print(f"GPU: L4 (24GB VRAM)")
    print(f"{'='*60}\n")

    run = await flyte.run.aio(
        process_arxiv_papers,
        papers,
        model_name="deepseek-ai/DeepSeek-OCR",
        batch_size=2,  # Optimal for DeepSeek-OCR
        dpi=200
    )

    print(f"\n📊 Pipeline execution URL: {run.url}")
    print(f"Execution name: {run.name}")

    # Wait for completion and get results
    await run.wait()
    results = run.outputs()

    print(f"\n✅ Pipeline completed!")
    print(f"📁 Results saved to {len(results)} files")

    # Download and display first result
    if results:
        first_result = results[0]
        local_path = await first_result.to_local()
        with open(local_path, 'r') as f:
            ocr_result = json.load(f)

        print(f"\n📄 Sample result ({ocr_result['arxiv_id']}):")
        print(f"   Title: {ocr_result['title']}")
        print(f"   Pages: {ocr_result['num_pages']}")
        print(f"   Success: {ocr_result['success']}")
        print(f"   Processing time: {ocr_result['processing_time']:.2f}s")
        if ocr_result['success']:
            print(f"   Content preview: {ocr_result['content'][:200]}...")


async def example_single_paper():
    """Example processing a single paper with custom resources."""
    papers = [
        ArxivPaper(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf"
        ),
    ]

    # Override resources for a single high-priority paper
    custom_worker = process_single_paper.override(
        resources=flyte.Resources(
            cpu=8,
            memory="24Gi",
            gpu="L4:1",
        ),
        reusable="off"  # Don't reuse for one-off tasks
    )

    await flyte.init_from_config.aio()
    result = await flyte.run.aio(
        custom_worker,
        papers[0].arxiv_id,
        papers[0].title,
        papers[0].pdf_url,
        model_name="deepseek-ai/DeepSeek-OCR",
        batch_size=2
    )

    print(f"Result: {result.url}")
    await result.wait()
    print(f"OCR completed for {papers[0].title}")


if __name__ == "__main__":
    # Run with DeepSeek-OCR (3B params, L4 GPU)
    asyncio.run(main())

    # Or run single paper example
    # asyncio.run(example_single_paper())
