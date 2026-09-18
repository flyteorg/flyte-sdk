# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte",
#    "flyteplugins-mlflow",
#    "scikit-learn",
#    "xgboost",
#    "optuna",
#    "mlflow",
#    "joblib",
# ]
# ///
"""
Model Selection & Hyperparameter Tuning with MLflow

This example demonstrates a complete model selection pipeline that:

1. Loads the breast cancer dataset (binary classification)
2. Runs Optuna hyperparameter searches in parallel for three model families:
   - XGBoost
   - Random Forest
   - Logistic Regression
3. Logs every trial as a nested MLflow run (visible in the MLflow UI)
4. Compares the best model from each family
5. Registers the overall winner in the MLflow Model Registry

Uses the flyteplugins-mlflow integration for:
- @mlflow_run decorator for automatic run management
- Mlflow() link class for clickable MLflow UI links in the Flyte UI
- mlflow_config() for workflow-level tracking URI and link_host config

Each model family runs as its own Flyte task in parallel. Optuna handles the
inner HPO loop, and each trial creates a nested MLflow run under the parent.

Pair with examples/apps/mlflow_server.py to deploy a shared MLflow UI.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import optuna
from flyteplugins.mlflow import Mlflow, mlflow_config, mlflow_run
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

import flyte
import flyte.io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

env = flyte.TaskEnvironment(
    name="model-selection",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    image=flyte.Image.from_uv_script(__file__, name="model-selection"),
    env_vars={"GIT_PYTHON_REFRESH": "quiet"},
)


@dataclass
class ModelResult:
    """Summary of the best model found for a given family."""

    family: str
    best_score: float
    best_params: dict
    model_file: flyte.io.File


# ---------------------------------------------------------------------------
# Per-family HPO tasks
#
# Each task uses @mlflow_run(run_mode="nested") so its MLflow run appears as
# a child of the parent orchestrator run in the MLflow UI.
# ---------------------------------------------------------------------------


@mlflow_run(run_mode="nested")
@env.task(links=(Mlflow(),))
async def tune_xgboost(
    n_trials: int = 20,
) -> ModelResult:
    """Run Optuna HPO for XGBoost and log every trial to MLflow."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    best = {"score": -1.0, "params": {}, "model": None}

    mlflow.log_param("model_family", "xgboost")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = XGBClassifier(**params, eval_metric="logloss", verbosity=0)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        mean_score = float(scores.mean())

        mlflow.log_metrics(
            {"cv_accuracy_mean": mean_score, "cv_accuracy_std": float(scores.std())},
            step=trial.number,
        )

        if mean_score > best["score"]:
            best["score"] = mean_score
            best["params"] = params
            model.fit(X, y)
            best["model"] = model

        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    mlflow.log_params({f"best_{k}": v for k, v in best["params"].items()})
    mlflow.log_metric("best_cv_accuracy", best["score"])

    # Save the best model as a Flyte file artifact
    model_path = Path("/tmp/best_xgboost.joblib")
    joblib.dump(best["model"], model_path)
    model_file = await flyte.io.File.from_local(str(model_path))

    logger.info(f"XGBoost best CV accuracy: {best['score']:.4f}")
    return ModelResult(
        family="xgboost",
        best_score=best["score"],
        best_params=best["params"],
        model_file=model_file,
    )


@mlflow_run(run_mode="nested")
@env.task(links=(Mlflow(),))
async def tune_random_forest(
    n_trials: int = 20,
) -> ModelResult:
    """Run Optuna HPO for Random Forest and log every trial to MLflow."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    best = {"score": -1.0, "params": {}, "model": None}

    mlflow.log_param("model_family", "random_forest")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        model = RandomForestClassifier(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        mean_score = float(scores.mean())

        mlflow.log_metrics(
            {"cv_accuracy_mean": mean_score, "cv_accuracy_std": float(scores.std())},
            step=trial.number,
        )

        if mean_score > best["score"]:
            best["score"] = mean_score
            best["params"] = params
            model.fit(X, y)
            best["model"] = model

        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    mlflow.log_params({f"best_{k}": v for k, v in best["params"].items()})
    mlflow.log_metric("best_cv_accuracy", best["score"])

    model_path = Path("/tmp/best_random_forest.joblib")
    joblib.dump(best["model"], model_path)
    model_file = await flyte.io.File.from_local(str(model_path))

    logger.info(f"Random Forest best CV accuracy: {best['score']:.4f}")
    return ModelResult(
        family="random_forest",
        best_score=best["score"],
        best_params=best["params"],
        model_file=model_file,
    )


@mlflow_run(run_mode="nested")
@env.task(links=(Mlflow(),))
async def tune_logistic_regression(
    n_trials: int = 20,
) -> ModelResult:
    """Run Optuna HPO for Logistic Regression and log every trial to MLflow."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    best = {"score": -1.0, "params": {}, "model": None}

    mlflow.log_param("model_family", "logistic_regression")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", 0.001, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "saga",
            "max_iter": 5000,
        }
        model = LogisticRegression(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        mean_score = float(scores.mean())

        mlflow.log_metrics(
            {"cv_accuracy_mean": mean_score, "cv_accuracy_std": float(scores.std())},
            step=trial.number,
        )

        if mean_score > best["score"]:
            best["score"] = mean_score
            best["params"] = params
            model.fit(X, y)
            best["model"] = model

        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    mlflow.log_params({f"best_{k}": v for k, v in best["params"].items()})
    mlflow.log_metric("best_cv_accuracy", best["score"])

    model_path = Path("/tmp/best_logistic_regression.joblib")
    joblib.dump(best["model"], model_path)
    model_file = await flyte.io.File.from_local(str(model_path))

    logger.info(f"Logistic Regression best CV accuracy: {best['score']:.4f}")
    return ModelResult(
        family="logistic_regression",
        best_score=best["score"],
        best_params=best["params"],
        model_file=model_file,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@mlflow_run
@env.task(links=(Mlflow(),))
async def model_selection_pipeline(
    n_trials: int = 20,
    experiment_name: str = "flyte-model-selection",
    registry_name: str = "breast-cancer-classifier",
) -> ModelResult:
    """
    Run HPO for all model families in parallel, pick the winner, and register
    the best model in the MLflow Model Registry.

    Configure the MLflow tracking URI and UI link host via mlflow_config()
    when launching the run (see __main__ below).
    """
    mlflow.log_params({"n_trials": n_trials, "experiment_name": experiment_name})

    # Fan out: each model family tunes in parallel on its own Flyte task.
    # Each creates a nested MLflow run under this parent.
    xgb_result, rf_result, lr_result = await asyncio.gather(
        tune_xgboost(n_trials),
        tune_random_forest(n_trials),
        tune_logistic_regression(n_trials),
    )

    # Compare results
    results = [xgb_result, rf_result, lr_result]
    for r in results:
        logger.info(f"  {r.family:25s}  CV accuracy = {r.best_score:.4f}")

    winner = max(results, key=lambda r: r.best_score)
    logger.info(f"Winner: {winner.family} with accuracy {winner.best_score:.4f}")

    # Download the winning model artifact and register it in MLflow
    local_path = await winner.model_file.download("/tmp/winning_model.joblib")
    model = joblib.load(local_path)

    # Log the model artifact using the appropriate MLflow flavor
    if winner.family == "xgboost":
        mlflow.xgboost.log_model(model, artifact_path="model")
    else:
        mlflow.sklearn.log_model(model, artifact_path="model")

    mlflow.log_param("winner_family", winner.family)
    mlflow.log_metric("winner_cv_accuracy", winner.best_score)

    # Register in the Model Registry
    from flyteplugins.mlflow import get_mlflow_run

    run_id = get_mlflow_run().info.run_id
    registered = mlflow.register_model(f"runs:/{run_id}/model", registry_name)
    logger.info(f"Registered '{registry_name}' version {registered.version} from run {run_id}")

    return winner


if __name__ == "__main__":
    flyte.init_from_config()

    # Set your deployed MLflow server URL here (see examples/apps/mlflow_server.py).
    # link_host generates clickable MLflow UI links in the Flyte UI.
    MLFLOW_SERVER_URL = "https://cold-bird-3fc86.apps.demo.hosted.unionai.cloud"

    run = flyte.with_runcontext(
        custom_context=mlflow_config(
            tracking_uri=MLFLOW_SERVER_URL,
            experiment_name="flyte-model-selection",
            link_host=MLFLOW_SERVER_URL,
        ),
    ).run(
        model_selection_pipeline,
        n_trials=20,
        experiment_name="flyte-model-selection",
        registry_name="breast-cancer-classifier",
    )
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")
