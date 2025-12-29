"""
OCR Processor - Class-based OCR processing with model caching

This module provides a clean interface for OCR processing:
- OCRProcessor class that loads and caches models
- Simple methods for single document and batch processing
- Automatic GPU/CPU detection and device management
"""

import asyncio
import logging
from typing import Any

import flyte.io
import pyarrow as pa
import torch
from async_lru import alru_cache
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    OCR Processor with cached model loading.

    This class encapsulates all OCR processing logic:
    - Loads the model once during initialization
    - Provides methods for single and batch document processing
    - Handles GPU/CPU placement automatically
    - Manages errors gracefully per document
    """

    def __init__(self, model_id: str):
        """
        Initialize the OCR processor and load the model.

        The model is loaded once during initialization and reused
        for all subsequent processing calls.

        Args:
            model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-VL-2B-Instruct")
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing OCRProcessor for model: {model_id}")
        logger.info(f"Using device: {self.device}")

        # Load processor/tokenizer
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        self.model.eval()
        logger.info(f"Model {model_id} loaded successfully on {self.device}")

    def run_inference(
        self,
        image: Image.Image,
        prompt: str = "Extract all text from this document image.",
    ) -> dict[str, Any]:
        """
        Run OCR inference on a single image.

        Args:
            image: PIL Image to process
            prompt: Instruction prompt for the model

        Returns:
            Dictionary with:
                - text: Extracted text
                - success: Boolean indicating success
                - error: Error message if failed (None if successful)
                - token_count: Number of tokens generated
        """
        try:
            # Prepare inputs with chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template and prepare inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                )

            # Decode output
            generated_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids, strict=True)
            ]
            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return {
                "text": output_text,
                "success": True,
                "error": None,
                "token_count": len(generated_ids[0]),
            }

        except Exception as e:
            logger.error(f"OCR inference failed: {e}")
            return {
                "text": "",
                "success": False,
                "error": str(e),
                "token_count": 0,
            }

    async def process_single_document(
        self,
        image_file: flyte.io.File,
        prompt: str = "Extract all text from this document image.",
    ) -> dict[str, Any]:
        """
        Process a single document file.

        Args:
            image_file: Flyte File object containing the document image
            prompt: OCR instruction prompt

        Returns:
            Dictionary with OCR results including document_id, text, success, error
        """
        doc_id = image_file.path if hasattr(image_file, "path") else str(image_file)

        try:
            # Download the file
            downloaded_path = await image_file.download()

            # Load image
            image = Image.open(downloaded_path).convert("RGB")

            # Run OCR
            result = self.run_inference(image, prompt)

            return {
                "document_id": doc_id,
                "extracted_text": result["text"],
                "success": result["success"],
                "error": result["error"],
                "token_count": result["token_count"],
            }

        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {e}")
            return {
                "document_id": doc_id,
                "extracted_text": "",
                "success": False,
                "error": str(e),
                "token_count": 0,
            }

    async def process_batch(
        self,
        model_name: str,
        image_files: list[flyte.io.File],
        batch_size: int = 4,
    ) -> flyte.io.DataFrame:
        """
        Process a batch of document images.

        Images are downloaded and processed in mini-batches for efficiency.
        Each document's result is tracked independently, so failures don't
        stop the entire batch.

        Args:
            model_name: Model name for tracking in results
            image_files: List of Flyte File objects to process
            batch_size: Number of images to download and process at once

        Returns:
            DataFrame with columns: document_id, model, extracted_text,
            success, error, token_count
        """
        logger.info(f"Processing batch of {len(image_files)} documents with {self.model_id}")

        # Result lists for DataFrame
        document_ids = []
        extracted_texts = []
        successes = []
        errors = []
        token_counts = []

        # Process images in mini-batches
        for i in range(0, len(image_files), batch_size):
            mini_batch_files = image_files[i : i + batch_size]

            # Download mini-batch files in parallel
            download_tasks = [file.download() for file in mini_batch_files]
            downloaded_paths = await asyncio.gather(*download_tasks, return_exceptions=True)

            # Process each image
            for file, downloaded_path in zip(mini_batch_files, downloaded_paths, strict=True):
                # Get document ID from file path
                doc_id = file.path if hasattr(file, "path") else str(file)

                # Handle download errors
                if isinstance(downloaded_path, Exception):
                    logger.warning(f"Failed to download {doc_id}: {downloaded_path}")
                    document_ids.append(doc_id)
                    extracted_texts.append("")
                    successes.append(False)
                    errors.append(str(downloaded_path))
                    token_counts.append(0)
                    continue

                # Load and process image
                try:
                    image = Image.open(downloaded_path).convert("RGB")

                    # Run OCR
                    result = self.run_inference(image)

                    document_ids.append(doc_id)
                    extracted_texts.append(result["text"])
                    successes.append(result["success"])
                    errors.append(result["error"])
                    token_counts.append(result["token_count"])

                    logger.info(f"Processed {doc_id}: {result['token_count']} tokens extracted")

                except Exception as e:
                    logger.error(f"Failed to process {doc_id}: {e}")
                    document_ids.append(doc_id)
                    extracted_texts.append("")
                    successes.append(False)
                    errors.append(str(e))
                    token_counts.append(0)

        # Create PyArrow table
        table = pa.table(
            {
                "document_id": document_ids,
                "model": [model_name] * len(document_ids),
                "extracted_text": extracted_texts,
                "success": successes,
                "error": errors,
                "token_count": token_counts,
            }
        )

        logger.info(f"Completed batch: {len(document_ids)} documents processed")
        return flyte.io.DataFrame.from_df(table)


# Async LRU cache for OCRProcessor instances
# This ensures each model is loaded only once and reused across all tasks


@alru_cache(maxsize=4)  # Cache up to 4 different models
async def get_ocr_processor(model_id: str) -> OCRProcessor:
    """
    Get or create a cached OCRProcessor instance.

    This function uses async LRU cache to ensure each model is loaded
    only once per worker container. Subsequent calls with the same model_id
    return the cached instance.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Cached OCRProcessor instance
    """
    logger.info(f"Creating OCRProcessor instance for {model_id}")
    return OCRProcessor(model_id)
