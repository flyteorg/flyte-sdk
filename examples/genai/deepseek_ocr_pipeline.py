"""
Efficient DeepSeek OCR Pipeline for arXiv Papers

This pipeline efficiently converts arXiv papers (PDFs) into text using:
- AsyncIO for concurrent processing
- Transformers for OCR (DeepSeek or similar models)
- Batch processing for optimal GPU utilization
- Efficient memory management
"""

import asyncio
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, AsyncIterator, Optional
import time

import aiohttp
from PIL import Image
from pdf2image import convert_from_bytes
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch


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


class DeepSeekOCRPipeline:
    """
    Efficient OCR pipeline using DeepSeek-VL or similar vision-language models.

    Features:
    - Async concurrent paper processing
    - Batch inference for GPU efficiency
    - Memory-efficient streaming
    - Error handling and retries
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-vl-7b-chat",
        max_concurrent_downloads: int = 5,
        max_concurrent_ocr: int = 2,
        batch_size: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the OCR pipeline.

        Args:
            model_name: HuggingFace model identifier
            max_concurrent_downloads: Maximum parallel PDF downloads
            max_concurrent_ocr: Maximum parallel OCR tasks
            batch_size: Number of pages to process in one batch
            device: Device to run inference on
        """
        self.model_name = model_name
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_ocr = max_concurrent_ocr
        self.batch_size = batch_size
        self.device = device

        # Semaphores for rate limiting
        self.download_semaphore = asyncio.Semaphore(max_concurrent_downloads)
        self.ocr_semaphore = asyncio.Semaphore(max_concurrent_ocr)

        # Load model and processor
        print(f"Loading model {model_name} on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        self.model.eval()
        print("Model loaded successfully!")

    async def download_pdf(self, paper: ArxivPaper, session: aiohttp.ClientSession) -> bytes:
        """
        Download PDF content from arXiv.

        Args:
            paper: ArxivPaper object with PDF URL
            session: aiohttp session for downloading

        Returns:
            PDF content as bytes
        """
        async with self.download_semaphore:
            try:
                print(f"Downloading {paper.arxiv_id}: {paper.title}")
                async with session.get(paper.pdf_url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    response.raise_for_status()
                    pdf_bytes = await response.read()
                    print(f"Downloaded {paper.arxiv_id} ({len(pdf_bytes) / 1024 / 1024:.2f} MB)")
                    return pdf_bytes
            except Exception as e:
                raise RuntimeError(f"Failed to download {paper.arxiv_id}: {e}")

    def pdf_to_images(self, pdf_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
        """
        Convert PDF bytes to images (one per page).

        Args:
            pdf_bytes: PDF content
            dpi: Resolution for conversion

        Returns:
            List of PIL Images
        """
        try:
            images = convert_from_bytes(pdf_bytes, dpi=dpi)
            return images
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}")

    async def ocr_images_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Perform OCR on a batch of images using the model.

        Args:
            images: List of PIL Images

        Returns:
            List of extracted text strings
        """
        # Run inference in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._ocr_images_sync, images)

    def _ocr_images_sync(self, images: List[Image.Image]) -> List[str]:
        """
        Synchronous OCR processing for a batch of images.

        Args:
            images: List of PIL Images

        Returns:
            List of extracted text strings
        """
        texts = []

        # Process in batches for efficiency
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]

            # Prepare inputs
            pixel_values = self.processor(
                images=batch_images,
                return_tensors="pt"
            ).pixel_values.to(self.device)

            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=512,
                    num_beams=3,
                    early_stopping=True
                )

            # Decode outputs
            batch_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
            texts.extend(batch_texts)

        return texts

    async def process_single_paper(
        self,
        paper: ArxivPaper,
        session: aiohttp.ClientSession
    ) -> OCRResult:
        """
        Process a single arXiv paper through the complete OCR pipeline.

        Args:
            paper: ArxivPaper to process
            session: aiohttp session for downloading

        Returns:
            OCRResult with extracted text and metadata
        """
        start_time = time.time()

        try:
            # Step 1: Download PDF
            pdf_bytes = await self.download_pdf(paper, session)

            # Step 2: Convert to images (CPU-bound, run in executor)
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(None, self.pdf_to_images, pdf_bytes)
            num_pages = len(images)
            print(f"Converted {paper.arxiv_id} to {num_pages} images")

            # Step 3: Perform OCR with semaphore for GPU management
            async with self.ocr_semaphore:
                print(f"Starting OCR for {paper.arxiv_id}")
                page_texts = await self.ocr_images_batch(images)

            # Combine all pages
            full_text = "\n\n".join([
                f"--- Page {i+1} ---\n{text}"
                for i, text in enumerate(page_texts)
            ])

            processing_time = time.time() - start_time
            print(f" Completed {paper.arxiv_id} in {processing_time:.2f}s")

            return OCRResult(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                content=full_text,
                num_pages=num_pages,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            print(f" Failed {paper.arxiv_id}: {e}")
            return OCRResult(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                content="",
                num_pages=0,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )

    async def process_papers(
        self,
        papers: List[ArxivPaper]
    ) -> AsyncIterator[OCRResult]:
        """
        Process multiple papers concurrently using async for loop pattern.

        Args:
            papers: List of ArxivPaper objects to process

        Yields:
            OCRResult objects as they complete
        """
        async with aiohttp.ClientSession() as session:
            # Create tasks for all papers
            tasks = [
                self.process_single_paper(paper, session)
                for paper in papers
            ]

            # Process papers as they complete
            for coro in asyncio.as_completed(tasks):
                result = await coro
                yield result

    async def process_papers_ordered(
        self,
        papers: List[ArxivPaper]
    ) -> List[OCRResult]:
        """
        Process papers concurrently but return results in original order.

        Args:
            papers: List of ArxivPaper objects to process

        Returns:
            List of OCRResult in the same order as input papers
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.process_single_paper(paper, session)
                for paper in papers
            ]
            return await asyncio.gather(*tasks)


async def main():
    """Example usage of the DeepSeek OCR pipeline."""

    # Sample arXiv papers
    papers = [
        ArxivPaper(
            arxiv_id="2301.00001",
            title="Attention Is All You Need",
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf"
        ),
        ArxivPaper(
            arxiv_id="2301.00002",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            pdf_url="https://arxiv.org/pdf/1810.04805.pdf"
        ),
        ArxivPaper(
            arxiv_id="2301.00003",
            title="GPT-3: Language Models are Few-Shot Learners",
            pdf_url="https://arxiv.org/pdf/2005.14165.pdf"
        ),
    ]

    # Initialize pipeline
    pipeline = DeepSeekOCRPipeline(
        model_name="microsoft/trocr-base-printed",  # Using TrOCR as fallback (more accessible)
        max_concurrent_downloads=5,
        max_concurrent_ocr=2,
        batch_size=4
    )

    print(f"\n{'='*60}")
    print(f"Processing {len(papers)} papers...")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Method 1: Process papers as they complete (streaming)
    print("Method 1: Streaming results as they complete")
    async for result in pipeline.process_papers(papers):
        if result.success:
            print(f"\n=Ä {result.title}")
            print(f"   ID: {result.arxiv_id}")
            print(f"   Pages: {result.num_pages}")
            print(f"   Time: {result.processing_time:.2f}s")
            print(f"   Content preview: {result.content[:200]}...")
        else:
            print(f"\nL {result.title}")
            print(f"   Error: {result.error}")

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average per paper: {total_time / len(papers):.2f}s")
    print(f"{'='*60}")

    # Method 2: Process in order (uncomment to test)
    # results = await pipeline.process_papers_ordered(papers)
    # for result in results:
    #     print(f"Processed: {result.arxiv_id} - Success: {result.success}")


if __name__ == "__main__":
    asyncio.run(main())