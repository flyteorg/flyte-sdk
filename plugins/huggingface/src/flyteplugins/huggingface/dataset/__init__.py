import functools

from ._source import HFSource, from_hf
from ._transformers import (
    HFToHuggingFaceDatasetDecodingHandler,
    HFToHuggingFaceIterableDatasetDecodingHandler,
    HuggingFaceDatasetToParquetEncodingHandler,
    HuggingFaceIterableDatasetToParquetEncodingHandler,
    ParquetToHuggingFaceDatasetDecodingHandler,
    ParquetToHuggingFaceIterableDatasetDecodingHandler,
)

__all__ = ["HFSource", "from_hf"]


@functools.lru_cache(maxsize=None)
def register_huggingface_dataset_transformers():
    """Register Hugging Face Dataset encoders and decoders."""
    from flyte.io.extend import DataFrameTransformerEngine

    DataFrameTransformerEngine.register(HuggingFaceDatasetToParquetEncodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register(ParquetToHuggingFaceDatasetDecodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register(HFToHuggingFaceDatasetDecodingHandler())
    DataFrameTransformerEngine.register(
        HuggingFaceIterableDatasetToParquetEncodingHandler(),
        default_format_for_type=True,
    )
    DataFrameTransformerEngine.register(
        ParquetToHuggingFaceIterableDatasetDecodingHandler(),
        default_format_for_type=True,
    )
    DataFrameTransformerEngine.register(HFToHuggingFaceIterableDatasetDecodingHandler())


register_huggingface_dataset_transformers()
