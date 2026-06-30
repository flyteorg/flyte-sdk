"""
Example: Using @mlflow_run(autolog=True) for automatic logging.

This example demonstrates:
- Autologging with sklearn (framework-specific) via mlflow_run(autolog=True)
- Autologging with generic mlflow.autolog()
- Configuring autolog via mlflow_config(autolog=True) context
- log_models and log_datasets control
- Auto-generated UI links via link_host config and Mlflow link class
"""

import logging
from pathlib import Path

import flyte
import numpy as np
from flyte._image import PythonWheels

from flyteplugins.mlflow import Mlflow, get_mlflow_run, mlflow_config, mlflow_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

DATABRICKS_USERNAME = "<username>"
DATABRICKS_HOST = "<host>"

env = flyte.TaskEnvironment(
    name="mlflow-autolog-example",
    image=flyte.Image.from_debian_base(name="mlflow-autolog-example")
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-mlflow",
            pre=True,
        ),
    )
    .with_pip_packages("scikit-learn", "numpy", "mlflow[databricks]"),
    secrets=[
        flyte.Secret(key="databricks_token", as_env_var="DATABRICKS_TOKEN"),
    ],
    env_vars={
        "MLFLOW_TRACKING_URI": "databricks",
        "GIT_PYTHON_REFRESH": "quiet",
        "DATABRICKS_HOST": DATABRICKS_HOST,
    },
)


# Framework-specific autolog with sklearn
@mlflow_run(autolog=True, framework="sklearn")
@env.task(links=(Mlflow(),))  # needs to be empty if you want the plugin to auto-generate
async def train_sklearn_model(n_samples: int = 100) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X = np.random.randn(n_samples, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model — autolog captures parameters, metrics, and model artifact
    model = LogisticRegression(max_iter=200, C=1.0)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")


# Generic autolog with log_models/log_datasets control
@mlflow_run(autolog=True, log_models=True, log_datasets=True)
@env.task(links=(Mlflow(),))
async def train_with_generic_autolog(n_samples: int = 100) -> None:
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.randn(n_samples, 4)
    y = (X[:, 0] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=10, max_depth=3)
    model.fit(X, y)

    print("Training complete with generic autolog")


# Autolog configured via mlflow_config(autolog=True) context — no decorator args needed
@mlflow_run
@env.task
async def train_with_context_autolog(n_samples: int = 100) -> None:
    from sklearn.tree import DecisionTreeClassifier

    X = np.random.randn(n_samples, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)

    print("Training complete with context-based autolog")


# Parent task that initializes the MLflow run and calls child autolog tasks.
# The link auto-propagates to children via link_host in mlflow_config().
@mlflow_run
@env.task
async def run_autolog_examples(n_samples: int = 200) -> None:
    run = get_mlflow_run()

    await train_sklearn_model(n_samples=n_samples)
    await train_with_generic_autolog(n_samples=n_samples)

    with mlflow_config(
        autolog=True,
        run_mode="new",
        experiment_id=run.info.experiment_id,  # same experiment, new run
        autolog_kwargs={"log_input_examples": True},
    ):
        await train_with_context_autolog.override(
            links=(
                Mlflow(
                    # hardcoding the link since the run ID is only available after the run starts
                    link=f"{DATABRICKS_HOST}/ml/experiments/{run.info.experiment_id}"
                ),
            )
        )(n_samples=n_samples)


if __name__ == "__main__":
    flyte.init_from_config()

    print("\n=== Running autolog examples via parent task ===")
    run = flyte.with_runcontext(
        custom_context=mlflow_config(
            experiment_name=f"/Users/{DATABRICKS_USERNAME}/autolog-parent",
            link_host=DATABRICKS_HOST,
            link_template="{host}/ml/experiments/{experiment_id}/runs/{run_id}",
        ),
    ).run(run_autolog_examples, n_samples=200)
    print(f"Run URL: {run.url}")
