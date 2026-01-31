"""
Example demonstrating NamedTuples with complex types in Flyte tasks.

This example shows how to use NamedTuples containing complex types like:
- Dataclasses
- Pydantic models
- flyte.io.File
- flyte.io.Dir
- flyte.io.DataFrame

NamedTuples with complex types provide a convenient way to return multiple
named values with rich type information from tasks.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import List, NamedTuple

import pandas as pd
from pydantic import BaseModel

import flyte
from flyte.io import DataFrame, Dir, File

env = flyte.TaskEnvironment(
    name="namedtuple_types_complex_example",
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


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a model."""

    precision: float
    recall: float
    f1_score: float


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


class ArtifactInfo(BaseModel):
    """Information about an artifact."""

    artifact_type: str
    size_bytes: int
    checksum: str


# ============================================================================
# NamedTuple Definitions with Complex Types
# ============================================================================


class TrainingOutputs(NamedTuple):
    """Outputs from model training containing dataclasses."""

    config: ModelConfig
    result: TrainingResult
    metrics: EvaluationMetrics


class ExperimentOutputs(NamedTuple):
    """Outputs from an experiment containing Pydantic models."""

    metadata: DatasetMetadata
    config: ExperimentConfig
    artifact_info: ArtifactInfo


class FileOutputs(NamedTuple):
    """Outputs containing File objects."""

    model_weights: File
    training_logs: File
    predictions: File


class DirOutputs(NamedTuple):
    """Outputs containing Dir objects."""

    training_data: Dir
    model_artifacts: Dir
    experiment_name: str


class DataFrameOutputs(NamedTuple):
    """Outputs containing DataFrame objects."""

    train_data: DataFrame
    test_data: DataFrame
    predictions: DataFrame


class MixedComplexOutputs(NamedTuple):
    """Outputs mixing multiple complex types."""

    config: ModelConfig
    metadata: DatasetMetadata
    model_file: File
    data_dir: Dir
    results_df: DataFrame


class NestedComplexOutputs(NamedTuple):
    """Outputs with nested NamedTuples containing complex types."""

    training: TrainingOutputs
    files: FileOutputs
    summary: str


# ============================================================================
# Tasks with NamedTuple[Dataclass, ...] Types
# ============================================================================


@env.task
async def create_training_outputs() -> TrainingOutputs:
    """Create a NamedTuple containing dataclass instances."""
    config = ModelConfig(
        model_name="deep_neural_network",
        learning_rate=0.001,
        epochs=100,
        hidden_layers=[256, 128, 64],
    )
    result = TrainingResult(accuracy=0.94, loss=0.06, trained_epochs=100)
    metrics = EvaluationMetrics(precision=0.93, recall=0.95, f1_score=0.94)

    return TrainingOutputs(config=config, result=result, metrics=metrics)


@env.task
async def process_training_outputs(outputs: TrainingOutputs) -> str:
    """Process a NamedTuple containing dataclass instances."""
    return (
        f"Model '{outputs.config.model_name}' trained for {outputs.result.trained_epochs} epochs: "
        f"accuracy={outputs.result.accuracy:.2%}, F1={outputs.metrics.f1_score:.2%}"
    )


# ============================================================================
# Tasks with NamedTuple[Pydantic, ...] Types
# ============================================================================


@env.task
async def create_experiment_outputs() -> ExperimentOutputs:
    """Create a NamedTuple containing Pydantic model instances."""
    metadata = DatasetMetadata(
        name="customer_churn",
        num_rows=10000,
        num_columns=15,
        feature_names=["tenure", "monthly_charges", "contract_type", "payment_method"],
    )
    config = ExperimentConfig(
        experiment_id="churn-exp-001",
        description="Customer churn prediction baseline",
        hyperparameters={"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"},
    )
    artifact_info = ArtifactInfo(
        artifact_type="model",
        size_bytes=15728640,
        checksum="sha256:abc123def456",
    )

    return ExperimentOutputs(metadata=metadata, config=config, artifact_info=artifact_info)


@env.task
async def process_experiment_outputs(outputs: ExperimentOutputs) -> str:
    """Process a NamedTuple containing Pydantic model instances."""
    return (
        f"Experiment '{outputs.config.experiment_id}' on '{outputs.metadata.name}': "
        f"{outputs.metadata.num_rows} rows, artifact size: {outputs.artifact_info.size_bytes / 1024 / 1024:.1f}MB"
    )


# ============================================================================
# Tasks with NamedTuple[File, ...] Types
# ============================================================================


@env.task
async def create_file_outputs() -> FileOutputs:
    """Create a NamedTuple containing File objects."""
    # Create model weights file
    weights_file = File.new_remote()
    async with weights_file.open("wb") as fh:
        await fh.write(b"Binary model weights data..." * 100)

    # Create training logs file
    logs_file = File.new_remote()
    async with logs_file.open("wb") as fh:
        log_content = "\n".join([f"Epoch {i}: loss={0.5 - i * 0.005:.4f}" for i in range(100)])
        await fh.write(log_content.encode("utf-8"))

    # Create predictions file
    predictions_file = File.new_remote()
    async with predictions_file.open("wb") as fh:
        await fh.write(b"id,prediction,confidence\n1,0,0.85\n2,1,0.92\n3,0,0.78\n")

    return FileOutputs(model_weights=weights_file, training_logs=logs_file, predictions=predictions_file)


@env.task
async def process_file_outputs(outputs: FileOutputs) -> dict:
    """Process a NamedTuple containing File objects."""
    sizes = {}

    async with outputs.model_weights.open("rb") as fh:
        sizes["weights"] = len(bytes(await fh.read()))

    async with outputs.training_logs.open("rb") as fh:
        sizes["logs"] = len(bytes(await fh.read()))

    async with outputs.predictions.open("rb") as fh:
        sizes["predictions"] = len(bytes(await fh.read()))

    print(f"File sizes: {sizes}")
    return sizes


# ============================================================================
# Tasks with NamedTuple[Dir, ...] Types
# ============================================================================


@env.task
async def create_dir_outputs() -> DirOutputs:
    """Create a NamedTuple containing Dir objects."""
    # Create training data directory
    train_dir = tempfile.mkdtemp(prefix="training_data_")
    with open(os.path.join(train_dir, "features.csv"), "w") as f:
        f.write("id,f1,f2,f3\n1,0.1,0.2,0.3\n2,0.4,0.5,0.6\n3,0.7,0.8,0.9\n")
    with open(os.path.join(train_dir, "labels.csv"), "w") as f:
        f.write("id,label\n1,0\n2,1\n3,0\n")
    os.makedirs(os.path.join(train_dir, "splits"))
    with open(os.path.join(train_dir, "splits", "train_ids.txt"), "w") as f:
        f.write("1\n2\n")
    with open(os.path.join(train_dir, "splits", "test_ids.txt"), "w") as f:
        f.write("3\n")
    training_data_dir = await Dir.from_local(train_dir)

    # Create model artifacts directory
    artifacts_dir = tempfile.mkdtemp(prefix="model_artifacts_")
    with open(os.path.join(artifacts_dir, "model_config.json"), "w") as f:
        f.write('{"model_type": "random_forest", "n_estimators": 100}')
    with open(os.path.join(artifacts_dir, "feature_importance.csv"), "w") as f:
        f.write("feature,importance\nf1,0.35\nf2,0.40\nf3,0.25\n")
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        f.write('{"accuracy": 0.92, "auc": 0.95}')
    model_artifacts_dir = await Dir.from_local(artifacts_dir)

    return DirOutputs(
        training_data=training_data_dir,
        model_artifacts=model_artifacts_dir,
        experiment_name="churn_prediction_v2",
    )


@env.task
async def process_dir_outputs(outputs: DirOutputs) -> dict:
    """Process a NamedTuple containing Dir objects."""
    file_counts = {"training_data": 0, "model_artifacts": 0}

    async for _ in outputs.training_data.walk(recursive=True):
        file_counts["training_data"] += 1

    async for _ in outputs.model_artifacts.walk(recursive=True):
        file_counts["model_artifacts"] += 1

    print(f"Experiment '{outputs.experiment_name}' file counts: {file_counts}")
    return file_counts


# ============================================================================
# Tasks with NamedTuple[DataFrame, ...] Types
# ============================================================================


@env.task
async def create_dataframe_outputs() -> DataFrameOutputs:
    """Create a NamedTuple containing DataFrame objects."""
    # Create training dataframe
    train_df = pd.DataFrame(
        {
            "customer_id": list(range(1, 101)),
            "tenure": [i * 2 for i in range(1, 101)],
            "monthly_charges": [50 + i * 0.5 for i in range(100)],
            "churn": [i % 3 == 0 for i in range(100)],
        }
    )
    train_fdf = DataFrame.from_df(train_df)

    # Create test dataframe
    test_df = pd.DataFrame(
        {
            "customer_id": list(range(101, 131)),
            "tenure": [i * 2 for i in range(30)],
            "monthly_charges": [60 + i * 0.5 for i in range(30)],
            "churn": [i % 4 == 0 for i in range(30)],
        }
    )
    test_fdf = DataFrame.from_df(test_df)

    # Create predictions dataframe
    predictions_df = pd.DataFrame(
        {
            "customer_id": list(range(101, 131)),
            "predicted_churn": [i % 3 == 0 for i in range(30)],
            "confidence": [0.7 + (i % 10) * 0.03 for i in range(30)],
        }
    )
    predictions_fdf = DataFrame.from_df(predictions_df)

    return DataFrameOutputs(train_data=train_fdf, test_data=test_fdf, predictions=predictions_fdf)


@env.task
async def process_dataframe_outputs(outputs: DataFrameOutputs) -> dict:
    """Process a NamedTuple containing DataFrame objects."""
    train_pd: pd.DataFrame = await outputs.train_data.open(pd.DataFrame).all()
    test_pd: pd.DataFrame = await outputs.test_data.open(pd.DataFrame).all()
    predictions_pd: pd.DataFrame = await outputs.predictions.open(pd.DataFrame).all()

    stats = {
        "train_rows": train_pd.shape[0],
        "test_rows": test_pd.shape[0],
        "predictions_rows": predictions_pd.shape[0],
        "total_rows": train_pd.shape[0] + test_pd.shape[0],
    }
    print(f"DataFrame stats: {stats}")
    return stats


# ============================================================================
# Tasks with Mixed Complex Types in NamedTuples
# ============================================================================


@env.task
async def create_mixed_complex_outputs() -> MixedComplexOutputs:
    """Create a NamedTuple mixing dataclass, pydantic, File, Dir, and DataFrame."""
    config = ModelConfig(
        model_name="ensemble_classifier",
        learning_rate=0.01,
        epochs=150,
        hidden_layers=[512, 256, 128],
    )

    metadata = DatasetMetadata(
        name="multimodal_dataset",
        num_rows=50000,
        num_columns=25,
        feature_names=[f"feature_{i}" for i in range(25)],
    )

    # Create model file
    model_file = File.new_remote()
    async with model_file.open("wb") as fh:
        await fh.write(b"Serialized ensemble model..." * 50)

    # Create data directory
    data_dir_path = tempfile.mkdtemp(prefix="data_")
    with open(os.path.join(data_dir_path, "metadata.json"), "w") as f:
        f.write('{"version": "2.0", "created": "2024-01-15"}')
    with open(os.path.join(data_dir_path, "schema.json"), "w") as f:
        f.write('{"fields": ["id", "features", "target"]}')
    data_dir = await Dir.from_local(data_dir_path)

    # Create results dataframe
    results_df = pd.DataFrame(
        {
            "model": ["rf", "xgb", "nn"],
            "accuracy": [0.91, 0.93, 0.95],
            "auc": [0.94, 0.96, 0.97],
        }
    )
    results_fdf = DataFrame.from_df(results_df)

    return MixedComplexOutputs(
        config=config,
        metadata=metadata,
        model_file=model_file,
        data_dir=data_dir,
        results_df=results_fdf,
    )


@env.task
async def process_mixed_complex_outputs(outputs: MixedComplexOutputs) -> str:
    """Process a NamedTuple with mixed complex types."""
    # Get file size
    async with outputs.model_file.open("rb") as fh:
        file_size = len(bytes(await fh.read()))

    # Count files in directory
    dir_file_count = 0
    async for _ in outputs.data_dir.walk():
        dir_file_count += 1

    # Get dataframe info
    results_pd: pd.DataFrame = await outputs.results_df.open(pd.DataFrame).all()
    best_model = results_pd.loc[results_pd["accuracy"].idxmax(), "model"]

    return (
        f"Model '{outputs.config.model_name}' on '{outputs.metadata.name}': "
        f"file={file_size}B, dir_files={dir_file_count}, best_model={best_model}"
    )


# ============================================================================
# Tasks with Nested NamedTuples of Complex Types
# ============================================================================


@env.task
async def create_nested_complex_outputs() -> NestedComplexOutputs:
    """Create nested NamedTuples with complex types."""
    # Create training outputs
    training = TrainingOutputs(
        config=ModelConfig(
            model_name="nested_deep_model",
            learning_rate=0.0005,
            epochs=300,
            hidden_layers=[1024, 512, 256, 128],
        ),
        result=TrainingResult(accuracy=0.97, loss=0.03, trained_epochs=300),
        metrics=EvaluationMetrics(precision=0.96, recall=0.98, f1_score=0.97),
    )

    # Create file outputs
    weights_file = File.new_remote()
    async with weights_file.open("wb") as fh:
        await fh.write(b"Deep model weights..." * 200)

    logs_file = File.new_remote()
    async with logs_file.open("wb") as fh:
        await fh.write(b"Training log entries..." * 100)

    predictions_file = File.new_remote()
    async with predictions_file.open("wb") as fh:
        await fh.write(b"Model predictions..." * 50)

    files = FileOutputs(
        model_weights=weights_file,
        training_logs=logs_file,
        predictions=predictions_file,
    )

    summary = (
        f"Nested outputs: model={training.config.model_name}, "
        f"accuracy={training.result.accuracy:.2%}, "
        f"files created successfully"
    )

    return NestedComplexOutputs(training=training, files=files, summary=summary)


@env.task
async def process_nested_complex_outputs(outputs: NestedComplexOutputs) -> str:
    """Process nested NamedTuples with complex types."""
    # Access nested training outputs
    model_name = outputs.training.config.model_name
    accuracy = outputs.training.result.accuracy
    f1 = outputs.training.metrics.f1_score

    # Calculate total file sizes
    total_size = 0
    async with outputs.files.model_weights.open("rb") as fh:
        total_size += len(bytes(await fh.read()))
    async with outputs.files.training_logs.open("rb") as fh:
        total_size += len(bytes(await fh.read()))
    async with outputs.files.predictions.open("rb") as fh:
        total_size += len(bytes(await fh.read()))

    return f"Model '{model_name}': accuracy={accuracy:.2%}, F1={f1:.2%}, total file size={total_size} bytes"


# ============================================================================
# Workflow Output NamedTuple
# ============================================================================


class WorkflowResults(NamedTuple):
    """Final workflow results."""

    training_summary: str
    experiment_summary: str
    file_sizes: dict
    dir_counts: dict
    df_stats: dict
    mixed_summary: str
    nested_summary: str


# ============================================================================
# Main Workflow
# ============================================================================


@env.task
async def complex_namedtuple_workflow() -> WorkflowResults:
    """Workflow demonstrating complex NamedTuple type usage."""
    print("=== Complex NamedTuple Types Workflow ===\n")

    # 1. Dataclass NamedTuples
    print("1. Processing dataclass NamedTuples...")
    training_outputs = await create_training_outputs()
    training_summary = await process_training_outputs(outputs=training_outputs)
    print(f"   Result: {training_summary}")

    # 2. Pydantic NamedTuples
    print("\n2. Processing Pydantic NamedTuples...")
    experiment_outputs = await create_experiment_outputs()
    experiment_summary = await process_experiment_outputs(outputs=experiment_outputs)
    print(f"   Result: {experiment_summary}")

    # 3. File NamedTuples
    print("\n3. Processing File NamedTuples...")
    file_outputs = await create_file_outputs()
    file_sizes = await process_file_outputs(outputs=file_outputs)
    print(f"   File sizes: {file_sizes}")

    # 4. Dir NamedTuples
    print("\n4. Processing Dir NamedTuples...")
    dir_outputs = await create_dir_outputs()
    dir_counts = await process_dir_outputs(outputs=dir_outputs)
    print(f"   File counts: {dir_counts}")

    # 5. DataFrame NamedTuples
    print("\n5. Processing DataFrame NamedTuples...")
    df_outputs = await create_dataframe_outputs()
    df_stats = await process_dataframe_outputs(outputs=df_outputs)
    print(f"   Stats: {df_stats}")

    # 6. Mixed complex NamedTuples
    print("\n6. Processing mixed complex NamedTuples...")
    mixed_outputs = await create_mixed_complex_outputs()
    mixed_summary = await process_mixed_complex_outputs(outputs=mixed_outputs)
    print(f"   Result: {mixed_summary}")

    # 7. Nested complex NamedTuples
    print("\n7. Processing nested complex NamedTuples...")
    nested_outputs = await create_nested_complex_outputs()
    nested_summary = await process_nested_complex_outputs(outputs=nested_outputs)
    print(f"   Result: {nested_summary}")

    print("\n=== Complex NamedTuple Types Workflow Completed! ===")

    return WorkflowResults(
        training_summary=training_summary,
        experiment_summary=experiment_summary,
        file_sizes=file_sizes,
        dir_counts=dir_counts,
        df_stats=df_stats,
        mixed_summary=mixed_summary,
        nested_summary=nested_summary,
    )


if __name__ == "__main__":
    flyte.init_from_config()

    print("Running complex NamedTuple types workflow...")
    run = flyte.run(complex_namedtuple_workflow)
    print(f"Run URL: {run.url}")
    run.wait()
    print("Complex NamedTuple workflow completed!")
    outputs = run.outputs()
    print(f"Outputs: {outputs}")
