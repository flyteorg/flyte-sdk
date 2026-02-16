# /// script
# requires-python = "==3.13"
# dependencies = [
#    "scikit-learn==1.6.1",
#    "numpy",
#    "pandas",
#    "pyarrow",
#    "joblib",
#    "mashumaro",
#    "flyte>=0.2.0b17",
# ]
# ///


import asyncio
import tempfile

import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import flyte
import flyte.errors
from flyte.io import Dir, File

env = flyte.TaskEnvironment(
    name="distributed_random_forest",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    image=flyte.Image.from_uv_script(
        __file__,
        name="flyte",
        registry="ghcr.io/flyteorg",
        platform=("linux/amd64", "linux/arm64"),
        python_version=(3, 13),
        pre=True,
    ).with_apt_packages("ca-certificates"),
)


# these constants are tuned such that the entire dataset is too large to fit into
# a machine with 250Mi of memory, but each partition is small enough to fit into
# memory.
N_SAMPLES = 20_000
N_CLASSES = 2
N_FEATURES = 10
N_INFORMATIVE = 5
N_REDUNDANT = 3
N_CLUSTERS_PER_CLASS = 1
FEATURE_NAMES = [f"feature_{i}" for i in range(N_FEATURES)]


@env.task
async def create_dataset(n_estimators: int) -> Dir:
    """Create a synthetic dataset that's too large to fit into memory, assuming 250Mi."""

    temp_dir = tempfile.mkdtemp()

    for i in range(n_estimators):
        print(f"Creating dataset {i}")
        X, y = make_classification(
            n_samples=N_SAMPLES,
            n_classes=N_CLASSES,
            n_features=N_FEATURES,
            n_informative=N_INFORMATIVE,
            n_redundant=N_REDUNDANT,
            n_clusters_per_class=N_CLUSTERS_PER_CLASS,
        )
        dataset = pd.DataFrame(X, columns=FEATURE_NAMES)
        dataset["target"] = y
        dataset.to_parquet(f"{temp_dir}/dataset_{i}.parquet")
        del X, y, dataset

    return await Dir.from_local(temp_dir)


@env.task
async def load_all_data(dataset_dir: Dir):
    """Try to load all the data into memory.

    This task demonstrates that loading the entire dataset into memory results
    in an out of memory error.
    """

    dataframes: list[pd.DataFrame] = []
    async for file in dataset_dir.walk():
        local_path = await file.download()
        df = pd.read_parquet(local_path)
        dataframes.append(df)

    data = pd.concat(dataframes)
    print(data.head())
    print(data.describe())


async def get_partition(dataset_dir: Dir, dataset_index: int) -> pd.DataFrame:
    """Helper function to get a partition of the dataset."""

    async for file in dataset_dir.walk():
        if file.name == f"dataset_{dataset_index}.parquet":
            local_path = await file.download()

    return pd.read_parquet(local_path)


@env.task
async def train_decision_tree(dataset_dir: Dir, dataset_index: int) -> File:
    """Train a decision tree on a subset of the dataset."""

    print(f"Training decision tree on partition {dataset_index}")
    dataset = await get_partition(dataset_dir, dataset_index)
    y = dataset["target"]
    X = dataset.drop(columns=["target"])
    model = DecisionTreeClassifier()
    model.fit(X, y)

    temp_dir = tempfile.mkdtemp()
    fp = f"{temp_dir}/decision_tree_{dataset_index}.joblib"
    joblib.dump(model, fp)
    return await File.from_local(fp)


async def load_decision_tree(file: File) -> DecisionTreeClassifier:
    local_path = await file.download()
    return joblib.load(local_path)


def random_forest_from_decision_trees(decision_trees: list[DecisionTreeClassifier]) -> RandomForestClassifier:
    """Helper function that reconstitutes a random forest model from a list of decision trees."""

    rf = RandomForestClassifier(n_estimators=len(decision_trees))
    rf.estimators_ = decision_trees
    rf.classes_ = decision_trees[0].classes_
    rf.n_classes_ = decision_trees[0].n_classes_
    rf.n_features_in_ = decision_trees[0].n_features_in_
    rf.n_outputs_ = decision_trees[0].n_outputs_
    rf.feature_names_in_ = FEATURE_NAMES
    return rf


@env.task
async def train_distributed_random_forest(dataset_dir: Dir, n_estimators: int) -> File:
    """Train a distributed random forest on the dataset.

    Random forest is an ensemble of decision trees that have been trained
    on subsets of a dataset. Here we implement distributed random forest where
    the full dataset cannot be loaded into memory. We therefore load partitions
    of the data into its own task and train decision tree on each partition.

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
    """Evaluate the random forest one partition of the dataset."""

    with random_forest.open_sync() as f:
        random_forest = joblib.load(f)

    data_partition = await get_partition(dataset_dir, dataset_index)
    y = data_partition["target"]
    X = data_partition.drop(columns=["target"])

    predictions = random_forest.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy: {accuracy}")
    return accuracy


@env.task
async def main(n_estimators: int = 16) -> tuple[File, float]:
    dataset = await create_dataset(n_estimators=n_estimators)
    try:
        await load_all_data(dataset)
    except flyte.errors.OOMError as e:
        print(f"Failed with oom trying with more resources: {e}, of type {type(e)}, {e.code}")

    random_forest = await train_distributed_random_forest(dataset, n_estimators)
    accuracy = await evaluate_random_forest(random_forest, dataset, 0)
    return random_forest, accuracy


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)

    # Run with:
    # uv run --prerelease=allow examples/ml/distributed_random_forest.py
