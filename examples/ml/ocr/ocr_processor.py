"""OCR Processor for Qwen Vision-Language Models"""

import logging
import os
from typing import Any

import flyte.io
import torch
from async_lru import alru_cache
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger(__name__)


class QwenOCRProcessor:
    """
    OCR processor using Qwen Vision-Language models.

    Loads the model once and reuses it for all processing.
    Automatically handles GPU/CPU placement.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """
        Initialize the OCR processor.

        Args:
            model_id: HuggingFace model ID for Qwen VL model
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading {model_id} on {self.device}")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=os.getenv("HF_HUB_TOKEN"),
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        self.model.eval()
        logger.info("Model loaded successfully")

    def extract_text(self, image: Image.Image, prompt: str = "Extract all text from this image.") -> dict[str, Any]:
        """
        Extract text from an image using OCR.

        Args:
            image: PIL Image to process
            prompt: Instruction prompt for the model

        Returns:
            Dictionary with text, success status, and token count
        """
        try:
            # Prepare input with chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate text
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)

            # Decode output
            generated_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids, strict=True)
            ]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return {
                "text": output_text,
                "success": True,
                "token_count": len(generated_ids[0]),
            }

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return {
                "text": "",
                "success": False,
                "token_count": 0,
            }

    async def process_document(self, image_file: flyte.io.File, prompt: str = "Extract all text from this image.") -> dict[str, Any]:
        """
        Process a single document file.

        Args:
            image_file: Flyte File object containing the image
            prompt: OCR instruction prompt

        Returns:
            Dictionary with document_id, extracted_text, success, token_count
        """
        doc_id = image_file.path if hasattr(image_file, "path") else str(image_file)

        try:
            downloaded_path = await image_file.download()
            image = Image.open(downloaded_path).convert("RGB")
            result = self.extract_text(image, prompt)

            return {
                "document_id": doc_id,
                "extracted_text": result["text"],
                "success": result["success"],
                "token_count": result["token_count"],
            }

        except Exception as e:
            logger.error(f"Failed to process {doc_id}: {e}")
            return {
                "document_id": doc_id,
                "extracted_text": "",
                "success": False,
                "token_count": 0,
            }


@alru_cache(maxsize=1)
async def get_ocr_processor(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> QwenOCRProcessor:
    """
    Get or create a cached OCR processor instance.

    The processor is loaded once and reused across all tasks.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Cached QwenOCRProcessor instance
    """
    return QwenOCRProcessor(model_id)
