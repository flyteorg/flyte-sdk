"""
Example: Hyperparameter tuning with Optuna and MLflow nested runs.

This example demonstrates:
- Using Optuna for hyperparameter optimization
- MLflow nested runs across Flyte tasks via run_mode="nested"
- Each trial runs as a separate Flyte task (supports heavy model training)
- Parent run tracks the overall study, child runs track individual trials
- Auto-generated UI links via link_host config and Mlflow link class
"""

import asyncio
from pathlib import Path

import flyte
import mlflow
import numpy as np
from flyte._image import PythonWheels
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from flyteplugins.mlflow import Mlflow, get_mlflow_run, mlflow_config, mlflow_run

DATABRICKS_USERNAME = "<username>"
DATABRICKS_HOST = "<host>"

env = flyte.TaskEnvironment(
    name="mlflow-hpo-example",
    image=flyte.Image.from_debian_base(name="mlflow-hpo-example")
    .clone(
        addl_layer=PythonWheels(
            wheel_dir=Path(__file__).parent.parent / "dist",
            package_name="flyteplugins-mlflow",
            pre=True,
        ),
    )
    .with_pip_packages("mlflow[databricks]", "scikit-learn", "numpy", "optuna"),
    secrets=[flyte.Secret(key="databricks_token", as_env_var="DATABRICKS_TOKEN")],
    env_vars={
        "MLFLOW_TRACKING_URI": "databricks",
        "GIT_PYTHON_REFRESH": "quiet",
        "DATABRICKS_HOST": DATABRICKS_HOST,
    },
)


async def _generate_data(n_samples: int = 1000):
    """Generate synthetic regression data."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, 8)
    y = X[:, 0] * 3 + X[:, 1] * 2 - X[:, 2] + 0.5 * X[:, 3] * X[:, 4] + rng.randn(n_samples) * 0.1
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Each trial is a Flyte task with run_mode="nested".
# This creates a new MLflow run nested under the parent's run via the
# mlflow.parentRunId tag — no need for the parent run to be active
# in the same process.
@mlflow_run(run_mode="nested")
@env.task(links=(Mlflow()))  # run ID isn't known yet, so the UI links to the parent run as "MLflow (parent)".
async def run_trial(
    trial_number: int,
    max_depth: int,
    n_estimators: int,
    max_features: float,
    min_samples_split: int,
    n_samples: int = 2000,
) -> float:
    """Run a single HPO trial as a Flyte task with a nested MLflow run."""
    run = get_mlflow_run()

    X_train, X_val, y_train, y_val = await _generate_data(n_samples=n_samples)

    params = {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "max_features": max_features,
        "min_samples_split": min_samples_split,
    }
    mlflow.log_params(params)
    mlflow.log_param("trial_number", trial_number)

    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = float(np.sqrt(mse))

    mlflow.log_metrics({"mse": mse, "rmse": rmse})

    print(f"Trial {trial_number}: rmse={rmse:.4f} | run_id={run.info.run_id} | params={params}")
    return rmse


# Parent task creates the study-level MLflow run.
# Child trial tasks nest under it via run_mode="nested".
@mlflow_run
@env.task(links=(Mlflow()))
async def hpo_optuna_search(n_trials: int = 30, n_samples: int = 2000) -> str:
    """Run Optuna HPO study with each trial as a nested Flyte task.

    The parent run tracks the overall study. Each trial runs as a separate
    Flyte task with run_mode="nested", creating a child run under the parent
    in the MLflow UI.
    """
    import optuna

    run = get_mlflow_run()
    print(f"HPO Parent Run ID: {run.info.run_id}")

    mlflow.log_params({"n_trials": n_trials, "search_method": "optuna_tpe"})

    # Optuna study for hyperparameter suggestions.
    # We batch trials: ask Optuna for a batch, run them in parallel
    # via asyncio.gather and then tell Optuna the results before the next batch.
    study = optuna.create_study(direction="minimize")
    batch_size = min(n_trials, 5)

    results = []
    for batch_start in range(0, n_trials, batch_size):
        batch_end = min(batch_start + batch_size, n_trials)

        # Ask Optuna for a batch of trial suggestions
        trials_and_params = []
        for i in range(batch_start, batch_end):
            trial = study.ask()
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 32),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
                "max_features": trial.suggest_float("max_features", 0.2, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }
            trials_and_params.append((i, trial, params))

        # Run all trials in the batch in parallel — each dispatches
        # to a separate Flyte task with run_mode="nested"
        rmses = await asyncio.gather(
            *(
                run_trial(
                    trial_number=i,
                    n_samples=n_samples,
                    **params,
                )
                for i, _, params in trials_and_params
            )
        )

        # Tell Optuna the results so it can inform the next batch
        for (i, trial, params), rmse in zip(trials_and_params, rmses):
            study.tell(trial, rmse)
            results.append({"trial": i, "params": params, "rmse": rmse})

    # Log best results to parent run
    mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})
    mlflow.log_metric("best_rmse", study.best_value)

    print(f"\nBest trial: {study.best_trial.params} -> rmse={study.best_value:.4f}")
    print(f"Total trials: {len(study.trials)}")

    return run.info.run_id


if __name__ == "__main__":
    flyte.init_from_config()

    print("=== Running Optuna HPO example ===")
    run = flyte.with_runcontext(
        custom_context=mlflow_config(
            experiment_name=f"/Users/{DATABRICKS_USERNAME}/hpo-optuna",
            tags={"search_type": "optuna"},
            link_host=DATABRICKS_HOST,
            link_template="{host}/ml/experiments/{experiment_id}/runs/{run_id}",
        ),
    ).run(hpo_optuna_search, n_trials=30, n_samples=2000)
    print(f"Run URL: {run.url}")
