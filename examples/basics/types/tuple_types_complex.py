"""
Example demonstrating typed tuples with complex types in Flyte tasks.

This example shows how to use typed tuples containing complex types like:
- Dataclasses
- Pydantic models
- flyte.io.File
- flyte.io.Dir
- flyte.io.DataFrame

These complex types can be nested within tuples and passed between tasks.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import List

import pandas as pd
from pydantic import BaseModel

import flyte
from flyte.io import DataFrame, Dir, File

env = flyte.TaskEnvironment(
    name="tuple_types_complex_example",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow", "pydantic"),
)


# ============================================================================
# Dataclass Definitions
# ============================================================================


@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""

    model_name: str
    learning_rate: float
    epochs: int
    hidden_layers: List[int]


@dataclass
class TrainingResult:
    """Results from model training."""

    accuracy: float
    loss: float
    trained_epochs: int


# ============================================================================
# Pydantic Model Definitions
# ============================================================================


class DatasetMetadata(BaseModel):
    """Metadata about a dataset."""

    name: str
    num_rows: int
    num_columns: int
    feature_names: List[str]


class ExperimentConfig(BaseModel):
    """Configuration for an ML experiment."""

    experiment_id: str
    description: str
    hyperparameters: dict


# ============================================================================
# Tasks with Tuple[Dataclass, ...] Types
# ============================================================================


@env.task
async def create_config_and_result() -> tuple[ModelConfig, TrainingResult]:
    """Create a tuple containing dataclass instances."""
    config = ModelConfig(
        model_name="neural_network",
        learning_rate=0.001,
        epochs=100,
        hidden_layers=[128, 64, 32],
    )
    result = TrainingResult(accuracy=0.95, loss=0.05, trained_epochs=100)
    return (config, result)


@env.task
async def process_config_result(data: tuple[ModelConfig, TrainingResult]) -> str:
    """Process a tuple containing dataclass instances."""
    config, result = data
    return (
        f"Model '{config.model_name}' trained for {result.trained_epochs} epochs "
        f"with accuracy {result.accuracy:.2%}"
    )


# ============================================================================
# Tasks with Tuple[Pydantic, ...] Types
# ============================================================================


@env.task
async def create_pydantic_tuple() -> tuple[DatasetMetadata, ExperimentConfig]:
    """Create a tuple containing Pydantic model instances."""
    metadata = DatasetMetadata(
        name="iris_dataset",
        num_rows=150,
        num_columns=5,
        feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
    )
    config = ExperimentConfig(
        experiment_id="exp-001",
        description="Baseline classification experiment",
        hyperparameters={"batch_size": 32, "optimizer": "adam"},
    )
    return (metadata, config)


@env.task
async def process_pydantic_tuple(data: tuple[DatasetMetadata, ExperimentConfig]) -> str:
    """Process a tuple containing Pydantic model instances."""
    metadata, config = data
    return (
        f"Experiment '{config.experiment_id}' on dataset '{metadata.name}' "
        f"({metadata.num_rows} rows, {metadata.num_columns} columns)"
    )


# ============================================================================
# Tasks with Tuple[File, ...] Types
# ============================================================================


@env.task
async def create_file_tuple() -> tuple[File, File, str]:
    """Create a tuple containing File objects."""
    # Create first file
    file1 = File.new_remote()
    async with file1.open("wb") as fh:
        await fh.write(b"Training data content here...")

    # Create second file
    file2 = File.new_remote()
    async with file2.open("wb") as fh:
        await fh.write(b"Model weights binary data...")

    return (file1, file2, "model_v1")


@env.task
async def process_file_tuple(data: tuple[File, File, str]) -> int:
    """Process a tuple containing File objects and return total size."""
    data_file, weights_file, version = data
    total_size = 0

    async with data_file.open("rb") as fh:
        content = await fh.read()
        total_size += len(bytes(content))

    async with weights_file.open("rb") as fh:
        content = await fh.read()
        total_size += len(bytes(content))

    print(f"Processed files for version '{version}', total size: {total_size} bytes")
    return total_size


# ============================================================================
# Tasks with Tuple[Dir, ...] Types
# ============================================================================


@env.task
async def create_dir_tuple() -> tuple[Dir, Dir, str]:
    """Create a tuple containing Dir objects."""
    # Create first directory with training data
    temp_dir1 = tempfile.mkdtemp(prefix="training_data_")
    with open(os.path.join(temp_dir1, "train.csv"), "w") as f:
        f.write("id,feature1,feature2,label\n1,0.5,0.3,1\n2,0.8,0.2,0\n")
    with open(os.path.join(temp_dir1, "test.csv"), "w") as f:
        f.write("id,feature1,feature2,label\n3,0.6,0.4,1\n")
    dir1 = await Dir.from_local(temp_dir1)

    # Create second directory with model artifacts
    temp_dir2 = tempfile.mkdtemp(prefix="model_artifacts_")
    with open(os.path.join(temp_dir2, "config.json"), "w") as f:
        f.write('{"model_type": "classifier", "version": "1.0"}')
    with open(os.path.join(temp_dir2, "metadata.txt"), "w") as f:
        f.write("Trained on: 2024-01-15\nAuthor: ML Team")
    dir2 = await Dir.from_local(temp_dir2)

    return (dir1, dir2, "experiment_001")


@env.task
async def process_dir_tuple(data: tuple[Dir, Dir, str]) -> int:
    """Process a tuple containing Dir objects and return total file count."""
    data_dir, artifacts_dir, experiment_id = data
    total_files = 0

    async for _ in data_dir.walk(recursive=True):
        total_files += 1

    async for _ in artifacts_dir.walk(recursive=True):
        total_files += 1

    print(f"Experiment '{experiment_id}' has {total_files} total files")
    return total_files


# ============================================================================
# Tasks with Tuple[DataFrame, ...] Types
# ============================================================================


@env.task
async def create_dataframe_tuple() -> tuple[DataFrame, DataFrame, str]:
    """Create a tuple containing DataFrame objects."""
    # Create training dataframe
    train_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "label": [0, 1, 0, 1, 0],
        }
    )
    train_fdf = DataFrame.from_df(train_df)

    # Create test dataframe
    test_df = pd.DataFrame(
        {
            "id": [6, 7, 8],
            "feature1": [0.6, 0.7, 0.8],
            "feature2": [6.0, 7.0, 8.0],
            "label": [1, 0, 1],
        }
    )
    test_fdf = DataFrame.from_df(test_df)

    return (train_fdf, test_fdf, "classification_v1")


@env.task
async def process_dataframe_tuple(data: tuple[DataFrame, DataFrame, str]) -> int:
    """Process a tuple containing DataFrame objects and return total row count."""
    train_df, test_df, dataset_name = data

    # Convert to pandas for processing
    train_pd: pd.DataFrame = await train_df.open(pd.DataFrame).all()  # type: ignore
    test_pd: pd.DataFrame = await test_df.open(pd.DataFrame).all()  # type: ignore

    total_rows = train_pd.shape[0] + test_pd.shape[0]
    print(f"Dataset '{dataset_name}' has {total_rows} total rows")
    return total_rows


# ============================================================================
# Tasks with Mixed Complex Types in Tuples
# ============================================================================


@env.task
async def create_mixed_complex_tuple() -> tuple[ModelConfig, DatasetMetadata, File, DataFrame]:
    """Create a tuple mixing dataclass, pydantic, File, and DataFrame."""
    config = ModelConfig(
        model_name="hybrid_model",
        learning_rate=0.01,
        epochs=50,
        hidden_layers=[256, 128],
    )

    metadata = DatasetMetadata(
        name="mixed_dataset",
        num_rows=1000,
        num_columns=10,
        feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "target"],
    )

    # Create a file with model notes
    notes_file = File.new_remote()
    async with notes_file.open("wb") as fh:
        await fh.write(b"Model training notes and observations...")

    # Create a dataframe with sample data
    sample_df = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "value": [10.5, 20.3, 30.1],
        }
    )
    sample_fdf = DataFrame.from_df(sample_df)

    return (config, metadata, notes_file, sample_fdf)


@env.task
async def process_mixed_complex_tuple(
    data: tuple[ModelConfig, DatasetMetadata, File, DataFrame],
) -> str:
    """Process a tuple with mixed complex types."""
    config, metadata, notes_file, sample_df = data

    # Read file content
    async with notes_file.open("rb") as fh:
        notes_content = bytes(await fh.read()).decode("utf-8")

    # Get dataframe info
    sample_pd: pd.DataFrame = await sample_df.open(pd.DataFrame).all()
    df_rows = sample_pd.shape[0]

    return (
        f"Config: {config.model_name}, "
        f"Dataset: {metadata.name} ({metadata.num_rows} rows), "
        f"Notes: {len(notes_content)} chars, "
        f"Samples: {df_rows} rows"
    )


# ============================================================================
# Tasks with Nested Tuples of Complex Types
# ============================================================================


@env.task
async def create_nested_complex_tuple() -> tuple[tuple[ModelConfig, TrainingResult], tuple[File, Dir]]:
    """Create nested tuples with complex types."""
    # Inner tuple 1: config and result
    config = ModelConfig(
        model_name="nested_model",
        learning_rate=0.005,
        epochs=200,
        hidden_layers=[512, 256, 128],
    )
    result = TrainingResult(accuracy=0.98, loss=0.02, trained_epochs=200)

    # Inner tuple 2: file and directory
    model_file = File.new_remote()
    async with model_file.open("wb") as fh:
        await fh.write(b"Serialized model data...")

    temp_dir = tempfile.mkdtemp(prefix="logs_")
    with open(os.path.join(temp_dir, "training_log.txt"), "w") as f:
        f.write("Epoch 1: loss=0.5\nEpoch 2: loss=0.3\n")
    logs_dir = await Dir.from_local(temp_dir)

    return ((config, result), (model_file, logs_dir))


@env.task
async def process_nested_complex_tuple(
    data: tuple[tuple[ModelConfig, TrainingResult], tuple[File, Dir]],
) -> str:
    """Process nested tuples with complex types."""
    (config, result), (model_file, logs_dir) = data

    # Get file size
    async with model_file.open("rb") as fh:
        file_size = len(bytes(await fh.read()))

    # Count log files
    log_count = 0
    async for _ in logs_dir.walk():
        log_count += 1

    return (
        f"Model '{config.model_name}' achieved {result.accuracy:.2%} accuracy, "
        f"model file: {file_size} bytes, log files: {log_count}"
    )


# ============================================================================
# Main Workflow
# ============================================================================


@env.task
async def complex_tuple_workflow() -> tuple[
    tuple[ModelConfig, TrainingResult],  # dataclass tuple
    tuple[DatasetMetadata, ExperimentConfig],  # pydantic tuple
    tuple[File, File, str],  # file tuple
    tuple[Dir, Dir, str],  # dir tuple
    tuple[DataFrame, DataFrame, str],  # dataframe tuple
    tuple[ModelConfig, DatasetMetadata, File, DataFrame],  # mixed complex tuple
    tuple[tuple[ModelConfig, TrainingResult], tuple[File, Dir]],  # nested complex tuple
]:
    """Workflow demonstrating complex tuple type usage with complex output types.

    This workflow returns tuples containing complex types as outputs, demonstrating
    that Flyte can serialize and deserialize complex nested structures including:
    - Dataclasses (ModelConfig, TrainingResult)
    - Pydantic models (DatasetMetadata, ExperimentConfig)
    - File objects
    - Dir objects
    - DataFrame objects
    - Nested tuples of all the above
    """
    print("=== Complex Tuple Types Workflow ===\n")

    # 1. Dataclass tuples
    print("1. Creating and processing dataclass tuples...")
    config_result = await create_config_and_result()
    dataclass_output = await process_config_result(data=config_result)
    print(f"   Processed result: {dataclass_output}")
    print(f"   Returning tuple: (ModelConfig, TrainingResult)")

    # 2. Pydantic tuples
    print("\n2. Creating and processing Pydantic tuples...")
    pydantic_tuple = await create_pydantic_tuple()
    pydantic_output = await process_pydantic_tuple(data=pydantic_tuple)
    print(f"   Processed result: {pydantic_output}")
    print(f"   Returning tuple: (DatasetMetadata, ExperimentConfig)")

    # 3. File tuples
    print("\n3. Creating and processing File tuples...")
    file_tuple = await create_file_tuple()
    file_output = await process_file_tuple(data=file_tuple)
    print(f"   Total file size: {file_output} bytes")
    print(f"   Returning tuple: (File, File, str)")

    # 4. Dir tuples
    print("\n4. Creating and processing Dir tuples...")
    dir_tuple = await create_dir_tuple()
    dir_output = await process_dir_tuple(data=dir_tuple)
    print(f"   Total files: {dir_output}")
    print(f"   Returning tuple: (Dir, Dir, str)")

    # 5. DataFrame tuples
    print("\n5. Creating and processing DataFrame tuples...")
    df_tuple = await create_dataframe_tuple()
    df_output = await process_dataframe_tuple(data=df_tuple)
    print(f"   Total rows: {df_output}")
    print(f"   Returning tuple: (DataFrame, DataFrame, str)")

    # 6. Mixed complex tuples
    print("\n6. Creating and processing mixed complex tuples...")
    mixed_tuple = await create_mixed_complex_tuple()
    mixed_output = await process_mixed_complex_tuple(data=mixed_tuple)
    print(f"   Processed result: {mixed_output}")
    print(f"   Returning tuple: (ModelConfig, DatasetMetadata, File, DataFrame)")

    # 7. Nested complex tuples
    print("\n7. Creating and processing nested complex tuples...")
    nested_tuple = await create_nested_complex_tuple()
    nested_output = await process_nested_complex_tuple(data=nested_tuple)
    print(f"   Processed result: {nested_output}")
    print(f"   Returning tuple: ((ModelConfig, TrainingResult), (File, Dir))")

    print("\n=== Complex Tuple Types Workflow Completed! ===")
    print("All complex tuple types are being returned as workflow outputs.")

    return (
        config_result,  # tuple[ModelConfig, TrainingResult]
        pydantic_tuple,  # tuple[DatasetMetadata, ExperimentConfig]
        file_tuple,  # tuple[File, File, str]
        dir_tuple,  # tuple[Dir, Dir, str]
        df_tuple,  # tuple[DataFrame, DataFrame, str]
        mixed_tuple,  # tuple[ModelConfig, DatasetMetadata, File, DataFrame]
        nested_tuple,  # tuple[tuple[ModelConfig, TrainingResult], tuple[File, Dir]]
    )


if __name__ == "__main__":
    flyte.init_from_config()

    print("Running complex tuple types workflow...")
    run = flyte.run(complex_tuple_workflow)
    print(f"Run URL: {run.url}")
    run.wait()
    print("Complex tuple workflow completed!")
    outputs = run.outputs()
    print(f"Outputs: {outputs}")
