# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte",
#    "scikit-learn",
#    "xgboost",
#    "optuna",
#    "mlflow",
#    "pandas",
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
3. Logs every trial to MLflow tracking (params, metrics)
4. Compares the best model from each family
5. Registers the overall winner in the MLflow Model Registry

Each model family runs as its own Flyte task, so they execute in parallel on
separate machines. Optuna handles the inner HPO loop within each task.

The best model from each family is returned as a serialized Flyte File artifact.
The orchestrator then re-logs the winner to MLflow and registers it — this
avoids cross-machine MLflow run ID references, since each task gets its own
local MLflow store.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import optuna
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
    # Point MLflow at a shared tracking server in prod:
    # secrets=[flyte.Secret(key="MLFLOW_TRACKING_URI", as_env_var="MLFLOW_TRACKING_URI")],
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
# ---------------------------------------------------------------------------


@env.task
async def tune_xgboost(
    experiment_name: str,
    n_trials: int = 20,
    mlflow_tracking_uri: str = "",
) -> ModelResult:
    """Run Optuna HPO for XGBoost and log every trial to MLflow."""
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    data = load_breast_cancer()
    X, y = data.data, data.target

    mlflow.set_experiment(experiment_name)
    best = {"score": -1.0, "params": {}, "model": None}

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

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_param("model_family", "xgboost")
            mlflow.log_metric("cv_accuracy_mean", mean_score)
            mlflow.log_metric("cv_accuracy_std", float(scores.std()))

        if mean_score > best["score"]:
            best["score"] = mean_score
            best["params"] = params
            model.fit(X, y)
            best["model"] = model

        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

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


@env.task
async def tune_random_forest(
    experiment_name: str,
    n_trials: int = 20,
    mlflow_tracking_uri: str = "",
) -> ModelResult:
    """Run Optuna HPO for Random Forest and log every trial to MLflow."""
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    data = load_breast_cancer()
    X, y = data.data, data.target

    mlflow.set_experiment(experiment_name)
    best = {"score": -1.0, "params": {}, "model": None}

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

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_param("model_family", "random_forest")
            mlflow.log_metric("cv_accuracy_mean", mean_score)
            mlflow.log_metric("cv_accuracy_std", float(scores.std()))

        if mean_score > best["score"]:
            best["score"] = mean_score
            best["params"] = params
            model.fit(X, y)
            best["model"] = model

        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

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


@env.task
async def tune_logistic_regression(
    experiment_name: str,
    n_trials: int = 20,
    mlflow_tracking_uri: str = "",
) -> ModelResult:
    """Run Optuna HPO for Logistic Regression and log every trial to MLflow."""
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    data = load_breast_cancer()
    X, y = data.data, data.target

    mlflow.set_experiment(experiment_name)
    best = {"score": -1.0, "params": {}, "model": None}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", 0.001, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "saga",  # supports both l1 and l2
            "max_iter": 5000,
        }
        model = LogisticRegression(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        mean_score = float(scores.mean())

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_param("model_family", "logistic_regression")
            mlflow.log_metric("cv_accuracy_mean", mean_score)
            mlflow.log_metric("cv_accuracy_std", float(scores.std()))

        if mean_score > best["score"]:
            best["score"] = mean_score
            best["params"] = params
            model.fit(X, y)
            best["model"] = model

        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

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


@env.task
async def model_selection_pipeline(
    n_trials: int = 20,
    experiment_name: str = "flyte-model-selection",
    registry_name: str = "breast-cancer-classifier",
    mlflow_tracking_uri: str = "",
) -> ModelResult:
    """
    Run HPO for all model families in parallel, pick the winner, and register
    the best model in the MLflow Model Registry.

    Set mlflow_tracking_uri to the endpoint of a deployed MLflow server
    (e.g. from examples/apps/mlflow_server.py) to get a shared UI for all
    experiments. Leave empty to use MLflow's default local file store.
    """
    # Fan out: each model family tunes in parallel on its own Flyte task
    xgb_result, rf_result, lr_result = await asyncio.gather(
        tune_xgboost(experiment_name, n_trials, mlflow_tracking_uri),
        tune_random_forest(experiment_name, n_trials, mlflow_tracking_uri),
        tune_logistic_regression(experiment_name, n_trials, mlflow_tracking_uri),
    )

    # Compare results
    results = [xgb_result, rf_result, lr_result]
    for r in results:
        logger.info(f"  {r.family:25s}  CV accuracy = {r.best_score:.4f}")

    winner = max(results, key=lambda r: r.best_score)
    logger.info(f"Winner: {winner.family} with accuracy {winner.best_score:.4f}")

    # Download the winning model artifact and register it in MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    local_path = await winner.model_file.download("/tmp/winning_model.joblib")
    model = joblib.load(local_path)

    with mlflow.start_run(run_name=f"winner-{winner.family}"):
        mlflow.log_params(winner.best_params)
        mlflow.log_param("model_family", winner.family)
        mlflow.log_metric("cv_accuracy_mean", winner.best_score)

        # Log the model artifact using the appropriate MLflow flavor
        if winner.family == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        # Register in the Model Registry
        run_id = mlflow.active_run().info.run_id
        registered = mlflow.register_model(f"runs:/{run_id}/model", registry_name)
        logger.info(f"Registered '{registry_name}' version {registered.version} from run {run_id}")

    return winner


if __name__ == "__main__":
    flyte.init_from_config()

    # Deploy the MLflow server app first (see examples/apps/mlflow_server.py),
    # then pass its endpoint here so all tasks log to the same shared store:
    #   mlflow_tracking_uri="https://<your-mlflow-app-endpoint>"
    run = flyte.run(
        model_selection_pipeline,
        n_trials=20,
        experiment_name="flyte-model-selection",
        registry_name="breast-cancer-classifier",
        mlflow_tracking_uri="",  # set to your deployed MLflow server URL
    )
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")
