"""
Example demonstrating TypedDicts with complex types in Flyte tasks.

This example shows how to use TypedDicts containing complex types like:
- Dataclasses
- Pydantic models
- Tuples
- NamedTuples
- dict
- list
- flyte.io.File
- flyte.io.Dir
- flyte.io.DataFrame

It also demonstrates types that contain TypedDicts, such as:
- Dataclasses with TypedDict fields
- Pydantic models with TypedDict fields
- NamedTuples with TypedDict elements
- Tuples with TypedDict elements
- Lists of TypedDicts
- Dicts with TypedDict values
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, TypedDict

import pandas as pd
from pydantic import BaseModel

import flyte
from flyte.io import DataFrame, Dir, File

env = flyte.TaskEnvironment(
    name="typeddict_types_complex_example",
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
# Basic TypedDict Definitions
# ============================================================================


class Coordinates(TypedDict):
    """Geographic coordinates."""

    latitude: float
    longitude: float
    altitude: float


class PersonInfo(TypedDict):
    """Information about a person."""

    name: str
    age: int
    email: str


# ============================================================================
# TypedDicts with Dataclass Types
# ============================================================================


class TrainingConfigOutput(TypedDict):
    """TypedDict containing dataclass instances."""

    config: ModelConfig
    result: TrainingResult
    metrics: EvaluationMetrics


@env.task
async def create_typeddict_with_dataclasses() -> TrainingConfigOutput:
    """Create a TypedDict containing dataclass instances."""
    config = ModelConfig(
        model_name="deep_neural_network",
        learning_rate=0.001,
        epochs=100,
        hidden_layers=[256, 128, 64],
    )
    result = TrainingResult(accuracy=0.94, loss=0.06, trained_epochs=100)
    metrics = EvaluationMetrics(precision=0.93, recall=0.95, f1_score=0.94)

    return TrainingConfigOutput(config=config, result=result, metrics=metrics)


@env.task
async def process_typeddict_with_dataclasses(data: TrainingConfigOutput) -> str:
    """Process a TypedDict containing dataclass instances."""
    return (
        f"Model '{data['config'].model_name}' trained for {data['result'].trained_epochs} epochs: "
        f"accuracy={data['result'].accuracy:.2%}, F1={data['metrics'].f1_score:.2%}"
    )


# ============================================================================
# TypedDicts with Pydantic Types
# ============================================================================


class ExperimentOutput(TypedDict):
    """TypedDict containing Pydantic model instances."""

    metadata: DatasetMetadata
    config: ExperimentConfig
    artifact_info: ArtifactInfo


@env.task
async def create_typeddict_with_pydantic() -> ExperimentOutput:
    """Create a TypedDict containing Pydantic model instances."""
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

    return ExperimentOutput(metadata=metadata, config=config, artifact_info=artifact_info)


@env.task
async def process_typeddict_with_pydantic(data: ExperimentOutput) -> str:
    """Process a TypedDict containing Pydantic model instances."""
    return (
        f"Experiment '{data['config'].experiment_id}' on '{data['metadata'].name}': "
        f"{data['metadata'].num_rows} rows, artifact size: {data['artifact_info'].size_bytes / 1024 / 1024:.1f}MB"
    )


# ============================================================================
# TypedDicts with Tuple Types
# ============================================================================


class TupleContainingTypedDict(TypedDict):
    """TypedDict containing tuple types."""

    coordinates_pair: tuple[Coordinates, Coordinates]
    metrics_with_label: tuple[float, float, float, str]
    nested_tuple: tuple[tuple[int, int], tuple[str, str]]


@env.task
async def create_typeddict_with_tuples() -> TupleContainingTypedDict:
    """Create a TypedDict containing tuple types."""
    coord1 = Coordinates(latitude=37.7749, longitude=-122.4194, altitude=16.0)
    coord2 = Coordinates(latitude=40.7128, longitude=-74.0060, altitude=10.0)

    return TupleContainingTypedDict(
        coordinates_pair=(coord1, coord2),
        metrics_with_label=(0.95, 0.92, 0.93, "excellent"),
        nested_tuple=((1, 2), ("start", "end")),
    )


@env.task
async def process_typeddict_with_tuples(data: TupleContainingTypedDict) -> str:
    """Process a TypedDict containing tuple types."""
    coord1, coord2 = data["coordinates_pair"]
    precision, recall, f1, label = data["metrics_with_label"]
    return (
        f"Route from ({coord1['latitude']:.2f}, {coord1['longitude']:.2f}) "
        f"to ({coord2['latitude']:.2f}, {coord2['longitude']:.2f}), "
        f"quality: {label} (F1={f1:.2%})"
    )


# ============================================================================
# TypedDicts with NamedTuple Types
# ============================================================================


class ModelMetrics(NamedTuple):
    """Named tuple for model metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float


class ExperimentResults(NamedTuple):
    """Named tuple for experiment results."""

    experiment_id: str
    model_name: str
    metrics: ModelMetrics


class NamedTupleContainingTypedDict(TypedDict):
    """TypedDict containing NamedTuple types."""

    results: ExperimentResults
    all_metrics: List[ModelMetrics]
    best_config: ModelConfig


@env.task
async def create_typeddict_with_namedtuples() -> NamedTupleContainingTypedDict:
    """Create a TypedDict containing NamedTuple types."""
    metrics = ModelMetrics(accuracy=0.96, precision=0.94, recall=0.97, f1_score=0.95)
    results = ExperimentResults(
        experiment_id="exp-001",
        model_name="random_forest",
        metrics=metrics,
    )
    all_metrics = [
        ModelMetrics(accuracy=0.92, precision=0.90, recall=0.93, f1_score=0.91),
        ModelMetrics(accuracy=0.94, precision=0.92, recall=0.95, f1_score=0.93),
        ModelMetrics(accuracy=0.96, precision=0.94, recall=0.97, f1_score=0.95),
    ]
    best_config = ModelConfig(
        model_name="random_forest",
        learning_rate=0.01,
        epochs=100,
        hidden_layers=[128, 64],
    )

    return NamedTupleContainingTypedDict(
        results=results,
        all_metrics=all_metrics,
        best_config=best_config,
    )


@env.task
async def process_typeddict_with_namedtuples(data: NamedTupleContainingTypedDict) -> str:
    """Process a TypedDict containing NamedTuple types."""
    results = data["results"]
    avg_accuracy = sum(m.accuracy for m in data["all_metrics"]) / len(data["all_metrics"])
    return (
        f"Experiment '{results.experiment_id}' with model '{results.model_name}': "
        f"final accuracy={results.metrics.accuracy:.2%}, avg accuracy={avg_accuracy:.2%}"
    )


# ============================================================================
# TypedDicts with Dict Types
# ============================================================================


class DictContainingTypedDict(TypedDict):
    """TypedDict containing dict types."""

    hyperparameters: Dict[str, float]
    feature_importance: Dict[str, float]
    model_versions: Dict[str, ModelConfig]
    nested_metadata: Dict[str, Dict[str, str]]


@env.task
async def create_typeddict_with_dicts() -> DictContainingTypedDict:
    """Create a TypedDict containing dict types."""
    return DictContainingTypedDict(
        hyperparameters={"learning_rate": 0.001, "dropout": 0.2, "batch_size": 32.0},
        feature_importance={"tenure": 0.35, "monthly_charges": 0.28, "contract_type": 0.22, "other": 0.15},
        model_versions={
            "v1": ModelConfig(model_name="baseline", learning_rate=0.01, epochs=50, hidden_layers=[64]),
            "v2": ModelConfig(model_name="improved", learning_rate=0.001, epochs=100, hidden_layers=[128, 64]),
        },
        nested_metadata={
            "author": {"name": "Alice", "email": "alice@example.com"},
            "reviewer": {"name": "Bob", "email": "bob@example.com"},
        },
    )


@env.task
async def process_typeddict_with_dicts(data: DictContainingTypedDict) -> str:
    """Process a TypedDict containing dict types."""
    lr = data["hyperparameters"]["learning_rate"]
    top_feature = max(data["feature_importance"].items(), key=lambda x: x[1])
    model_count = len(data["model_versions"])
    return f"LR={lr}, top feature: {top_feature[0]} ({top_feature[1]:.2%}), {model_count} model versions"


# ============================================================================
# TypedDicts with List Types
# ============================================================================


class ListContainingTypedDict(TypedDict):
    """TypedDict containing list types."""

    coordinates_list: List[Coordinates]
    metrics_history: List[EvaluationMetrics]
    nested_lists: List[List[int]]
    string_tags: List[str]


@env.task
async def create_typeddict_with_lists() -> ListContainingTypedDict:
    """Create a TypedDict containing list types."""
    coords = [
        Coordinates(latitude=37.7749, longitude=-122.4194, altitude=16.0),
        Coordinates(latitude=34.0522, longitude=-118.2437, altitude=89.0),
        Coordinates(latitude=40.7128, longitude=-74.0060, altitude=10.0),
    ]
    metrics_history = [
        EvaluationMetrics(precision=0.85, recall=0.82, f1_score=0.83),
        EvaluationMetrics(precision=0.88, recall=0.86, f1_score=0.87),
        EvaluationMetrics(precision=0.92, recall=0.90, f1_score=0.91),
    ]

    return ListContainingTypedDict(
        coordinates_list=coords,
        metrics_history=metrics_history,
        nested_lists=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        string_tags=["production", "v2.0", "stable", "optimized"],
    )


@env.task
async def process_typeddict_with_lists(data: ListContainingTypedDict) -> str:
    """Process a TypedDict containing list types."""
    num_coords = len(data["coordinates_list"])
    final_f1 = data["metrics_history"][-1].f1_score
    num_tags = len(data["string_tags"])
    return f"{num_coords} locations, final F1={final_f1:.2%}, {num_tags} tags: {', '.join(data['string_tags'])}"


# ============================================================================
# TypedDicts with File Types
# ============================================================================


class FileContainingTypedDict(TypedDict):
    """TypedDict containing File objects."""

    model_weights: File
    training_logs: File
    predictions_file: File
    experiment_name: str


@env.task
async def create_typeddict_with_files() -> FileContainingTypedDict:
    """Create a TypedDict containing File objects."""
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

    return FileContainingTypedDict(
        model_weights=weights_file,
        training_logs=logs_file,
        predictions_file=predictions_file,
        experiment_name="churn_prediction_v1",
    )


@env.task
async def process_typeddict_with_files(data: FileContainingTypedDict) -> dict:
    """Process a TypedDict containing File objects."""
    sizes = {}

    async with data["model_weights"].open("rb") as fh:
        sizes["weights"] = len(bytes(await fh.read()))

    async with data["training_logs"].open("rb") as fh:
        sizes["logs"] = len(bytes(await fh.read()))

    async with data["predictions_file"].open("rb") as fh:
        sizes["predictions"] = len(bytes(await fh.read()))

    print(f"Experiment '{data['experiment_name']}' file sizes: {sizes}")
    return sizes


# ============================================================================
# TypedDicts with Dir Types
# ============================================================================


class DirContainingTypedDict(TypedDict):
    """TypedDict containing Dir objects."""

    training_data: Dir
    model_artifacts: Dir
    experiment_name: str
    version: int


@env.task
async def create_typeddict_with_dirs() -> DirContainingTypedDict:
    """Create a TypedDict containing Dir objects."""
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
    model_artifacts_dir = await Dir.from_local(artifacts_dir)

    return DirContainingTypedDict(
        training_data=training_data_dir,
        model_artifacts=model_artifacts_dir,
        experiment_name="churn_prediction_v2",
        version=2,
    )


@env.task
async def process_typeddict_with_dirs(data: DirContainingTypedDict) -> dict:
    """Process a TypedDict containing Dir objects."""
    file_counts = {"training_data": 0, "model_artifacts": 0}

    async for _ in data["training_data"].walk(recursive=True):
        file_counts["training_data"] += 1

    async for _ in data["model_artifacts"].walk(recursive=True):
        file_counts["model_artifacts"] += 1

    print(f"Experiment '{data['experiment_name']}' v{data['version']} file counts: {file_counts}")
    return file_counts


# ============================================================================
# TypedDicts with DataFrame Types
# ============================================================================


class DataFrameContainingTypedDict(TypedDict):
    """TypedDict containing DataFrame objects."""

    train_data: DataFrame
    test_data: DataFrame
    predictions: DataFrame
    dataset_name: str


@env.task
async def create_typeddict_with_dataframes() -> DataFrameContainingTypedDict:
    """Create a TypedDict containing DataFrame objects."""
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

    return DataFrameContainingTypedDict(
        train_data=train_fdf,
        test_data=test_fdf,
        predictions=predictions_fdf,
        dataset_name="customer_churn",
    )


@env.task
async def process_typeddict_with_dataframes(data: DataFrameContainingTypedDict) -> dict:
    """Process a TypedDict containing DataFrame objects."""
    train_pd: pd.DataFrame = await data["train_data"].open(pd.DataFrame).all()
    test_pd: pd.DataFrame = await data["test_data"].open(pd.DataFrame).all()
    predictions_pd: pd.DataFrame = await data["predictions"].open(pd.DataFrame).all()

    stats = {
        "train_rows": train_pd.shape[0],
        "test_rows": test_pd.shape[0],
        "predictions_rows": predictions_pd.shape[0],
        "total_rows": train_pd.shape[0] + test_pd.shape[0],
        "dataset_name": data["dataset_name"],
    }
    print(f"DataFrame stats: {stats}")
    return stats


# ============================================================================
# TypedDicts with Mixed Complex Types
# ============================================================================


class MixedComplexTypedDict(TypedDict):
    """TypedDict mixing dataclass, pydantic, File, Dir, and DataFrame."""

    config: ModelConfig
    metadata: DatasetMetadata
    model_file: File
    data_dir: Dir
    results_df: DataFrame


@env.task
async def create_mixed_complex_typeddict() -> MixedComplexTypedDict:
    """Create a TypedDict mixing dataclass, pydantic, File, Dir, and DataFrame."""
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

    return MixedComplexTypedDict(
        config=config,
        metadata=metadata,
        model_file=model_file,
        data_dir=data_dir,
        results_df=results_fdf,
    )


@env.task
async def process_mixed_complex_typeddict(data: MixedComplexTypedDict) -> str:
    """Process a TypedDict with mixed complex types."""
    # Get file size
    async with data["model_file"].open("rb") as fh:
        file_size = len(bytes(await fh.read()))

    # Count files in directory
    dir_file_count = 0
    async for _ in data["data_dir"].walk():
        dir_file_count += 1

    # Get dataframe info
    results_pd: pd.DataFrame = await data["results_df"].open(pd.DataFrame).all()
    best_model = results_pd.loc[results_pd["accuracy"].idxmax(), "model"]

    return (
        f"Model '{data['config'].model_name}' on '{data['metadata'].name}': "
        f"file={file_size}B, dir_files={dir_file_count}, best_model={best_model}"
    )


# ============================================================================
# Types Containing TypedDicts - Dataclass with TypedDict Field
# ============================================================================


@dataclass
class DataclassWithTypedDict:
    """A dataclass containing a TypedDict field."""

    experiment_name: str
    coordinates: Coordinates
    person: PersonInfo
    score: float


@env.task
async def create_dataclass_with_typeddict() -> DataclassWithTypedDict:
    """Create a dataclass containing TypedDict fields."""
    return DataclassWithTypedDict(
        experiment_name="geo_experiment",
        coordinates=Coordinates(latitude=37.7749, longitude=-122.4194, altitude=16.0),
        person=PersonInfo(name="Alice", age=30, email="alice@example.com"),
        score=0.95,
    )


@env.task
async def process_dataclass_with_typeddict(data: DataclassWithTypedDict) -> str:
    """Process a dataclass containing TypedDict fields."""
    return (
        f"Experiment '{data.experiment_name}' by {data.person['name']}: "
        f"location ({data.coordinates['latitude']:.2f}, {data.coordinates['longitude']:.2f}), "
        f"score={data.score:.2%}"
    )


# ============================================================================
# Types Containing TypedDicts - Pydantic Model with TypedDict Field
# ============================================================================


class PydanticWithTypedDict(BaseModel):
    """A Pydantic model containing TypedDict fields."""

    model_config = {"arbitrary_types_allowed": True}

    project_name: str
    location: Coordinates
    owner: PersonInfo
    priority: int


@env.task
async def create_pydantic_with_typeddict() -> PydanticWithTypedDict:
    """Create a Pydantic model containing TypedDict fields."""
    return PydanticWithTypedDict(
        project_name="smart_city",
        location=Coordinates(latitude=34.0522, longitude=-118.2437, altitude=89.0),
        owner=PersonInfo(name="Bob", age=35, email="bob@example.com"),
        priority=1,
    )


@env.task
async def process_pydantic_with_typeddict(data: PydanticWithTypedDict) -> str:
    """Process a Pydantic model containing TypedDict fields."""
    return (
        f"Project '{data.project_name}' owned by {data.owner['name']}: "
        f"location ({data.location['latitude']:.2f}, {data.location['longitude']:.2f}), "
        f"priority={data.priority}"
    )


# ============================================================================
# Types Containing TypedDicts - NamedTuple with TypedDict Element
# ============================================================================


class NamedTupleWithTypedDict(NamedTuple):
    """A NamedTuple containing TypedDict elements."""

    task_name: str
    location: Coordinates
    assignee: PersonInfo
    status: str


@env.task
async def create_namedtuple_with_typeddict() -> NamedTupleWithTypedDict:
    """Create a NamedTuple containing TypedDict elements."""
    return NamedTupleWithTypedDict(
        task_name="data_collection",
        location=Coordinates(latitude=40.7128, longitude=-74.0060, altitude=10.0),
        assignee=PersonInfo(name="Charlie", age=28, email="charlie@example.com"),
        status="in_progress",
    )


@env.task
async def process_namedtuple_with_typeddict(data: NamedTupleWithTypedDict) -> str:
    """Process a NamedTuple containing TypedDict elements."""
    return (
        f"Task '{data.task_name}' assigned to {data.assignee['name']}: "
        f"location ({data.location['latitude']:.2f}, {data.location['longitude']:.2f}), "
        f"status={data.status}"
    )


# ============================================================================
# Types Containing TypedDicts - Tuple with TypedDict Elements
# ============================================================================


@env.task
async def create_tuple_with_typeddicts() -> tuple[Coordinates, PersonInfo, str]:
    """Create a tuple containing TypedDict elements."""
    return (
        Coordinates(latitude=51.5074, longitude=-0.1278, altitude=11.0),
        PersonInfo(name="Diana", age=32, email="diana@example.com"),
        "completed",
    )


@env.task
async def process_tuple_with_typeddicts(data: tuple[Coordinates, PersonInfo, str]) -> str:
    """Process a tuple containing TypedDict elements."""
    coords, person, status = data
    return (
        f"Person {person['name']} at ({coords['latitude']:.2f}, {coords['longitude']:.2f}), "
        f"status: {status}"
    )


# ============================================================================
# Types Containing TypedDicts - List of TypedDicts
# ============================================================================


@env.task
async def create_list_of_typeddicts() -> List[Coordinates]:
    """Create a list of TypedDict instances."""
    return [
        Coordinates(latitude=37.7749, longitude=-122.4194, altitude=16.0),
        Coordinates(latitude=34.0522, longitude=-118.2437, altitude=89.0),
        Coordinates(latitude=40.7128, longitude=-74.0060, altitude=10.0),
        Coordinates(latitude=51.5074, longitude=-0.1278, altitude=11.0),
    ]


@env.task
async def process_list_of_typeddicts(data: List[Coordinates]) -> str:
    """Process a list of TypedDict instances."""
    avg_lat = sum(c["latitude"] for c in data) / len(data)
    avg_lon = sum(c["longitude"] for c in data) / len(data)
    return f"{len(data)} locations, centroid: ({avg_lat:.2f}, {avg_lon:.2f})"


# ============================================================================
# Types Containing TypedDicts - Dict with TypedDict Values
# ============================================================================


@env.task
async def create_dict_of_typeddicts() -> Dict[str, PersonInfo]:
    """Create a dict with TypedDict values."""
    return {
        "admin": PersonInfo(name="Alice", age=30, email="alice@example.com"),
        "developer": PersonInfo(name="Bob", age=35, email="bob@example.com"),
        "analyst": PersonInfo(name="Charlie", age=28, email="charlie@example.com"),
    }


@env.task
async def process_dict_of_typeddicts(data: Dict[str, PersonInfo]) -> str:
    """Process a dict with TypedDict values."""
    roles = list(data.keys())
    names = [p["name"] for p in data.values()]
    return f"Roles: {', '.join(roles)}. Team: {', '.join(names)}"


# ============================================================================
# Nested TypedDicts with Complex Types
# ============================================================================


class NestedComplexTypedDict(TypedDict):
    """TypedDict with nested TypedDicts and complex types."""

    training_output: TrainingConfigOutput
    experiment_output: ExperimentOutput
    file_output: FileContainingTypedDict
    summary: str


@env.task
async def create_nested_complex_typeddict() -> NestedComplexTypedDict:
    """Create deeply nested TypedDicts with complex types."""
    # Create training config output
    training_output = TrainingConfigOutput(
        config=ModelConfig(
            model_name="nested_deep_model",
            learning_rate=0.0005,
            epochs=300,
            hidden_layers=[1024, 512, 256, 128],
        ),
        result=TrainingResult(accuracy=0.97, loss=0.03, trained_epochs=300),
        metrics=EvaluationMetrics(precision=0.96, recall=0.98, f1_score=0.97),
    )

    # Create experiment output
    experiment_output = ExperimentOutput(
        metadata=DatasetMetadata(
            name="nested_dataset",
            num_rows=100000,
            num_columns=50,
            feature_names=[f"f{i}" for i in range(50)],
        ),
        config=ExperimentConfig(
            experiment_id="nested-exp-001",
            description="Nested experiment with complex types",
            hyperparameters={"lr": 0.0005, "batch_size": 128},
        ),
        artifact_info=ArtifactInfo(
            artifact_type="model",
            size_bytes=31457280,
            checksum="sha256:xyz789abc123",
        ),
    )

    # Create file output
    weights_file = File.new_remote()
    async with weights_file.open("wb") as fh:
        await fh.write(b"Deep nested model weights..." * 200)

    logs_file = File.new_remote()
    async with logs_file.open("wb") as fh:
        await fh.write(b"Training log entries..." * 100)

    predictions_file = File.new_remote()
    async with predictions_file.open("wb") as fh:
        await fh.write(b"Model predictions..." * 50)

    file_output = FileContainingTypedDict(
        model_weights=weights_file,
        training_logs=logs_file,
        predictions_file=predictions_file,
        experiment_name="nested_churn_prediction",
    )

    summary = (
        f"Nested outputs: model={training_output['config'].model_name}, "
        f"accuracy={training_output['result'].accuracy:.2%}, "
        f"dataset={experiment_output['metadata'].name}"
    )

    return NestedComplexTypedDict(
        training_output=training_output,
        experiment_output=experiment_output,
        file_output=file_output,
        summary=summary,
    )


@env.task
async def process_nested_complex_typeddict(data: NestedComplexTypedDict) -> str:
    """Process deeply nested TypedDicts with complex types."""
    model_name = data["training_output"]["config"].model_name
    accuracy = data["training_output"]["result"].accuracy
    dataset_name = data["experiment_output"]["metadata"].name

    # Calculate total file sizes
    total_size = 0
    async with data["file_output"]["model_weights"].open("rb") as fh:
        total_size += len(bytes(await fh.read()))
    async with data["file_output"]["training_logs"].open("rb") as fh:
        total_size += len(bytes(await fh.read()))
    async with data["file_output"]["predictions_file"].open("rb") as fh:
        total_size += len(bytes(await fh.read()))

    return (
        f"Model '{model_name}' on '{dataset_name}': accuracy={accuracy:.2%}, "
        f"total file size={total_size} bytes"
    )


# ============================================================================
# Workflow Output TypedDict
# ============================================================================


class WorkflowResults(TypedDict):
    """Final workflow results."""

    dataclass_summary: str
    pydantic_summary: str
    tuple_summary: str
    namedtuple_summary: str
    dict_summary: str
    list_summary: str
    file_sizes: dict
    dir_counts: dict
    df_stats: dict
    mixed_summary: str
    dataclass_with_td: str
    pydantic_with_td: str
    namedtuple_with_td: str
    tuple_with_td: str
    list_of_td: str
    dict_of_td: str
    nested_summary: str


# ============================================================================
# Main Workflow
# ============================================================================


@env.task
async def complex_typeddict_workflow() -> WorkflowResults:
    """Workflow demonstrating complex TypedDict type usage."""
    print("=== Complex TypedDict Types Workflow ===\n")

    # 1. TypedDict with dataclasses
    print("1. Processing TypedDict with dataclasses...")
    dataclass_output = await create_typeddict_with_dataclasses()
    dataclass_summary = await process_typeddict_with_dataclasses(data=dataclass_output)
    print(f"   Result: {dataclass_summary}")

    # 2. TypedDict with Pydantic models
    print("\n2. Processing TypedDict with Pydantic models...")
    pydantic_output = await create_typeddict_with_pydantic()
    pydantic_summary = await process_typeddict_with_pydantic(data=pydantic_output)
    print(f"   Result: {pydantic_summary}")

    # 3. TypedDict with tuples
    print("\n3. Processing TypedDict with tuples...")
    tuple_output = await create_typeddict_with_tuples()
    tuple_summary = await process_typeddict_with_tuples(data=tuple_output)
    print(f"   Result: {tuple_summary}")

    # 4. TypedDict with NamedTuples
    print("\n4. Processing TypedDict with NamedTuples...")
    namedtuple_output = await create_typeddict_with_namedtuples()
    namedtuple_summary = await process_typeddict_with_namedtuples(data=namedtuple_output)
    print(f"   Result: {namedtuple_summary}")

    # 5. TypedDict with dicts
    print("\n5. Processing TypedDict with dicts...")
    dict_output = await create_typeddict_with_dicts()
    dict_summary = await process_typeddict_with_dicts(data=dict_output)
    print(f"   Result: {dict_summary}")

    # 6. TypedDict with lists
    print("\n6. Processing TypedDict with lists...")
    list_output = await create_typeddict_with_lists()
    list_summary = await process_typeddict_with_lists(data=list_output)
    print(f"   Result: {list_summary}")

    # 7. TypedDict with Files
    print("\n7. Processing TypedDict with Files...")
    file_output = await create_typeddict_with_files()
    file_sizes = await process_typeddict_with_files(data=file_output)
    print(f"   File sizes: {file_sizes}")

    # 8. TypedDict with Dirs
    print("\n8. Processing TypedDict with Dirs...")
    dir_output = await create_typeddict_with_dirs()
    dir_counts = await process_typeddict_with_dirs(data=dir_output)
    print(f"   File counts: {dir_counts}")

    # 9. TypedDict with DataFrames
    print("\n9. Processing TypedDict with DataFrames...")
    df_output = await create_typeddict_with_dataframes()
    df_stats = await process_typeddict_with_dataframes(data=df_output)
    print(f"   Stats: {df_stats}")

    # 10. Mixed complex TypedDict
    print("\n10. Processing mixed complex TypedDict...")
    mixed_output = await create_mixed_complex_typeddict()
    mixed_summary = await process_mixed_complex_typeddict(data=mixed_output)
    print(f"   Result: {mixed_summary}")

    # 11. Dataclass with TypedDict
    print("\n11. Processing dataclass with TypedDict...")
    dataclass_td = await create_dataclass_with_typeddict()
    dataclass_with_td = await process_dataclass_with_typeddict(data=dataclass_td)
    print(f"   Result: {dataclass_with_td}")

    # 12. Pydantic with TypedDict
    print("\n12. Processing Pydantic with TypedDict...")
    pydantic_td = await create_pydantic_with_typeddict()
    pydantic_with_td = await process_pydantic_with_typeddict(data=pydantic_td)
    print(f"   Result: {pydantic_with_td}")

    # 13. NamedTuple with TypedDict
    print("\n13. Processing NamedTuple with TypedDict...")
    namedtuple_td = await create_namedtuple_with_typeddict()
    namedtuple_with_td = await process_namedtuple_with_typeddict(data=namedtuple_td)
    print(f"   Result: {namedtuple_with_td}")

    # 14. Tuple with TypedDicts
    print("\n14. Processing tuple with TypedDicts...")
    tuple_td = await create_tuple_with_typeddicts()
    tuple_with_td = await process_tuple_with_typeddicts(data=tuple_td)
    print(f"   Result: {tuple_with_td}")

    # 15. List of TypedDicts
    print("\n15. Processing list of TypedDicts...")
    list_td = await create_list_of_typeddicts()
    list_of_td = await process_list_of_typeddicts(data=list_td)
    print(f"   Result: {list_of_td}")

    # 16. Dict of TypedDicts
    print("\n16. Processing dict of TypedDicts...")
    dict_td = await create_dict_of_typeddicts()
    dict_of_td = await process_dict_of_typeddicts(data=dict_td)
    print(f"   Result: {dict_of_td}")

    # 17. Nested complex TypedDicts
    print("\n17. Processing nested complex TypedDicts...")
    nested_output = await create_nested_complex_typeddict()
    nested_summary = await process_nested_complex_typeddict(data=nested_output)
    print(f"   Result: {nested_summary}")

    print("\n=== Complex TypedDict Types Workflow Completed! ===")

    return WorkflowResults(
        dataclass_summary=dataclass_summary,
        pydantic_summary=pydantic_summary,
        tuple_summary=tuple_summary,
        namedtuple_summary=namedtuple_summary,
        dict_summary=dict_summary,
        list_summary=list_summary,
        file_sizes=file_sizes,
        dir_counts=dir_counts,
        df_stats=df_stats,
        mixed_summary=mixed_summary,
        dataclass_with_td=dataclass_with_td,
        pydantic_with_td=pydantic_with_td,
        namedtuple_with_td=namedtuple_with_td,
        tuple_with_td=tuple_with_td,
        list_of_td=list_of_td,
        dict_of_td=dict_of_td,
        nested_summary=nested_summary,
    )


if __name__ == "__main__":
    flyte.init_from_config()

    print("Running complex TypedDict types workflow...")
    run = flyte.run(complex_typeddict_workflow)
    print(f"Run URL: {run.url}")
    run.wait()
    print("Complex TypedDict workflow completed!")
    outputs = run.outputs()
    print(f"Outputs: {outputs}")
