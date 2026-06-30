import asyncio
import tempfile

import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import flyte
from flyte.io import Dir, File

env = flyte.TaskEnvironment(name="distributed_random_forest")

N_FEATURES = 10


@env.task
async def create_dataset(n_estimators: int) -> Dir:
    """Create a synthetic dataset."""

    temp_dir = tempfile.mkdtemp()

    for i in range(n_estimators):
        print(f"Creating dataset {i}")
        x_values, y_values = make_classification(
            n_samples=1_000,
            n_classes=2,
            n_features=N_FEATURES,
            n_informative=5,
            n_redundant=3,
            n_clusters_per_class=1,
        )
        dataset = pd.DataFrame(x_values, columns=[f"feature_{i}" for i in range(N_FEATURES)])
        dataset["target"] = y_values
        dataset.to_parquet(f"{temp_dir}/dataset_{i}.parquet")
        del x_values, y_values, dataset

    return await Dir.from_local(temp_dir)


async def get_partition(dataset_dir: Dir, dataset_index: int) -> pd.DataFrame:
    """Helper function to get a partition of the dataset."""
    local_path = None
    async for file in dataset_dir.walk():
        if file.name == f"dataset_{dataset_index}.parquet":
            local_path = await file.download()
            break

    if local_path is None:
        raise FileNotFoundError(f"dataset_{dataset_index}.parquet not found")
    return pd.read_parquet(local_path)


@env.task
async def train_decision_tree(dataset_dir: Dir, dataset_index: int) -> File:
    """Train a decision tree on a subset of the dataset."""

    print(f"Training decision tree on partition {dataset_index}")
    dataset = await get_partition(dataset_dir, dataset_index)
    y_values = dataset["target"]
    x_values = dataset.drop(columns=["target"])
    model = DecisionTreeClassifier()
    model.fit(x_values, y_values)

    temp_dir = tempfile.mkdtemp()
    fp = f"{temp_dir}/decision_tree_{dataset_index}.joblib"
    joblib.dump(model, fp)
    return await File.from_local(fp)


async def load_decision_tree(file: File) -> DecisionTreeClassifier:
    local_path = await file.download()
    return joblib.load(local_path)


def random_forest_from_decision_trees(decision_trees: list[DecisionTreeClassifier]) -> RandomForestClassifier:
    """Helper function that reconstitutes a random forest from decision trees."""

    rf = RandomForestClassifier(n_estimators=len(decision_trees))
    rf.estimators_ = decision_trees
    rf.classes_ = decision_trees[0].classes_
    rf.n_classes_ = decision_trees[0].n_classes_
    rf.n_features_in_ = decision_trees[0].n_features_in_
    rf.n_outputs_ = decision_trees[0].n_outputs_
    rf.feature_names_in_ = [f"feature_{i}" for i in range(N_FEATURES)]
    return rf


@env.task
async def train_distributed_random_forest(dataset_dir: Dir, n_estimators: int) -> File:
    """Train a distributed random forest on the dataset.

    Random forest is an ensemble of decision trees that have been trained
    on subsets of a dataset. Here we implement distributed random forest where
    the full dataset cannot be loaded into memory. We therefore load partitions
    of the data into its own task and train a decision tree on each partition.

    After training, we reconstitute the random forest from the collection
    of trained decision tree models.
    """
    decision_tree_files: list[File] = []

    with flyte.group(f"parallel-training-{n_estimators}-decision-trees"):
        for i in range(n_estimators):
            decision_tree_files.append(train_decision_tree(dataset_dir, i))

        decision_tree_files = await asyncio.gather(*decision_tree_files)

    decision_trees = await asyncio.gather(*[load_decision_tree(file) for file in decision_tree_files])

    random_forest = random_forest_from_decision_trees(decision_trees)
    temp_dir = tempfile.mkdtemp()
    fp = f"{temp_dir}/random_forest.joblib"
    joblib.dump(random_forest, fp)
    return await File.from_local(fp)


@env.task
async def evaluate_random_forest(
    random_forest: File,
    dataset_dir: Dir,
    dataset_index: int,
) -> float:
    """Evaluate the random forest on one dataset partition."""

    with random_forest.open_sync() as file_handle:
        random_forest_model = joblib.load(file_handle)

    data_partition = await get_partition(dataset_dir, dataset_index)
    y_values = data_partition["target"]
    x_values = data_partition.drop(columns=["target"])

    predictions = random_forest_model.predict(x_values)
    accuracy = accuracy_score(y_values, predictions)
    print(f"Accuracy: {accuracy}")
    return accuracy


@env.task
async def main(n_estimators: int = 8) -> tuple[File, float]:
    dataset = await create_dataset(n_estimators=n_estimators)
    random_forest = await train_distributed_random_forest(dataset, n_estimators)
    accuracy = await evaluate_random_forest(random_forest, dataset, 0)
    return random_forest, accuracy
