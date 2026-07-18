"""
Data Engineering Pipeline: ETL for ML Workloads

This example demonstrates Flyte's data engineering capabilities for preprocessing
large multimodal datasets - a key use case mentioned in the SPEC.
"""

# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte",
#     "pandas>=2.0.0",
#     "pyarrow>=14.0.0",
# ]
# ///

import asyncio
from datetime import datetime
from typing import Any, Dict

import flyte
from flyte import Image


def get_data_engineering_image() -> Image:
    """Get image with data processing libraries."""
    return Image.from_debian_base(name="data-eng", python_version=(3, 12)).with_pip_packages(
        "flyte",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "polars>=0.20.0",
        "requests",
    )


image = get_data_engineering_image()
base_env = flyte.TaskEnvironment(name="data_eng", image=image)


@base_env.task
async def ingest_raw_data() -> Dict[str, Any]:
    """
    Step 1: Ingest raw data from various sources.

    Simulates ingesting millions of media files and metadata.
    """
    await asyncio.sleep(0.5)  # Simulate I/O

    return {
        "raw_data_sources": ["s3://bucket/audio", "s3://bucket/video", "api://metadata"],
        "records_ingested": 1_000_000,
        "data_types": ["audio", "video", "text_metadata"],
        "ingestion_timestamp": datetime.utcnow().isoformat(),
    }


@base_env.task
async def preprocess_audio(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Preprocess audio data.

    Transcodes, normalizes, and extracts features from audio files.
    """
    await asyncio.sleep(0.3)

    return {
        **raw_data,
        "preprocessing": {
            "audio_transcoded": True,
            "sample_rate_normalized": 16_000,
            "features_extracted": ["mfcc", "spectrogram", "zerocrossing"],
        },
        "processed_records": 950_000,  # Some records may be filtered
    }


@base_env.task
async def preprocess_video(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: Preprocess video data.

    Extracts frames, converts formats, extracts metadata.
    """
    await asyncio.sleep(0.4)

    return {
        **raw_data,
        "preprocessing": {
            "video_frames_extracted": True,
            "frame_rate_normalized": 30,
            "formats_converted": ["mp4", "webm"],
        },
        "processed_records": 980_000,
    }


@base_env.task
async def preprocess_text(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 4: Preprocess text metadata.

    Tokenizes, normalizes, and validates text data.
    """
    await asyncio.sleep(0.2)

    return {
        **raw_data,
        "preprocessing": {
            "tokenized": True,
            "lowercased": True,
            "special_chars_removed": True,
            "valid_samples": 995_000,
        },
    }


@base_env.task
async def filter_and_curate(
    audio_data: Dict[str, Any],
    video_data: Dict[str, Any],
    text_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Step 5: Filter and curate dataset.

    Removes low-quality samples, deduplicates, applies quality filters.
    """
    await asyncio.sleep(0.3)

    return {
        "filtered_audio": audio_data,
        "filtered_video": video_data,
        "filtered_text": text_data,
        "curated_records": 850_000,
        "quality_metrics": {
            "deduplication_rate": 0.12,
            "low_quality_filtered": 0.15,
            "final_quality_score": 0.93,
        },
    }


@base_env.task
async def extract_features(
    curated_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Step 6: Extract ML features.

    Computes embeddings and other features for model training.
    """
    await asyncio.sleep(0.4)

    return {
        **curated_data,
        "features": {
            "text_embeddings": "sentence-transformers/all-MiniLM-L6-v2",
            "audio_embeddings": "speech-resnet34",
            "video_features": ["frame_level", "clip_level", "scene_level"],
            "feature_dimensions": {"text": 384, "audio": 512, "video": 2048},
        },
        "feature_extraction_time_s": 120.5,
    }


@base_env.task
async def create_train_validation_split(
    feature_data: Dict[str, Any],
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    Step 7: Split data into train/validation/test sets.

    Ensures proper distribution across splits.
    """
    total_records = feature_data.get("curated_records", 850_000)

    return {
        **feature_data,
        "splits": {
            "train": int(total_records * train_ratio),
            "validation": int(total_records * (1 - train_ratio) / 2),
            "test": int(total_records * (1 - train_ratio) / 2),
        },
        "split_timestamp": datetime.utcnow().isoformat(),
    }


@base_env.task
async def prepare_dataset_for_training(
    split_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Step 8: Prepare final training dataset.

    Converts to efficient formats (Parquet/Arrow), indexes for fast access.
    """
    await asyncio.sleep(0.3)

    return {
        **split_data,
        "training_dataset": {
            "format": "parquet",
            "compression": "snappy",
            "partitioned_by": ["source", "quality_score"],
            "indexed": True,
        },
        "dataset_size_gb": 45.2,
    }


@base_env.task
async def log_data_quality(
    training_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Step 9: Log data quality metrics.

    Tracks dataset statistics for reproducibility and monitoring.
    """
    return {
        **training_data,
        "data_quality_log": {
            "total_records_processed": 1_000_000,
            "final_records": training_data.get("curated_records", 850_000),
            "quality_score": training_data.get("quality_metrics", {}).get("final_quality_score"),
            "features_count": len(training_data.get("features", {})),
            "logged_at": datetime.utcnow().isoformat(),
        },
    }


async def run_data_pipeline():
    """Run the complete data engineering pipeline."""
    flyte.init_from_config()

    print("=" * 60)
    print("Data Engineering Pipeline for ML Workloads")
    print("=" * 60)

    # Step 1: Ingest
    print("\n[Step 1] Ingesting raw data...")
    raw_data = await flyte.run(ingest_raw_data)
    print(f"Raw data ingested: {raw_data.outputs[0]}")

    # Steps 2-4: Preprocess each modality in parallel
    print("\n[Steps 2-4] Preprocessing modalities...")
    audio_result, video_result, text_result = await asyncio.gather(
        flyte.run(preprocess_audio, raw_data=raw_data.outputs[0]),
        flyte.run(preprocess_video, raw_data=raw_data.outputs[0]),
        flyte.run(preprocess_text, raw_data=raw_data.outputs[0]),
    )

    # Step 5: Filter and curate
    print("\n[Step 5] Filtering and curating...")
    curated = await flyte.run(
        filter_and_curate,
        audio_data=audio_result.outputs[0],
        video_data=video_result.outputs[0],
        text_data=text_result.outputs[0],
    )
    print(f"Curated: {curated.outputs[0]['curated_records']} records")

    # Step 6: Extract features
    print("\n[Step 6] Extracting features...")
    features = await flyte.run(extract_features, curated_data=curated.outputs[0])
    print(f"Features: {features.outputs[0]['features']}")

    # Step 7: Create splits
    print("\n[Step 7] Creating train/validation/test split...")
    splits = await flyte.run(
        create_train_validation_split,
        feature_data=features.outputs[0],
    )
    print(f"Split sizes: {splits.outputs[0]['splits']}")

    # Step 8: Prepare training dataset
    print("\n[Step 8] Preparing training dataset...")
    training = await flyte.run(prepare_dataset_for_training, split_data=splits.outputs[0])
    print(f"Dataset ready: {training.outputs[0]['training_dataset']}")

    # Step 9: Log quality
    print("\n[Step 9] Logging data quality...")
    quality_log = await flyte.run(log_data_quality, training_data=training.outputs[0])
    print(f"Quality log: {quality_log.outputs[0]['data_quality_log']}")


if __name__ == "__main__":
    import asyncio

    # asyncio.run(run_data_pipeline())

    print("Data Engineering Pipeline Example")
    print("==================================")
    print()
    print("Pipeline stages:")
    print("1. Ingest raw data from multiple sources")
    print("2-4. Preprocess audio, video, text modalities")
    print("5. Filter and curate dataset")
    print("6. Extract ML features (embeddings)")
    print("7. Create train/validation/test splits")
    print("8. Prepare training dataset (Parquet/Arrow)")
    print("9. Log data quality metrics")
