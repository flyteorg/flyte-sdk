# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "flyte",
#     "mlflow",
# ]
# ///
"""
MLflow Tracking Server

Deploys MLflow's built-in UI and tracking server as a Flyte app. Once deployed,
you get a DNS endpoint that any Flyte task can use as its MLFLOW_TRACKING_URI to
log experiments, metrics, and models to a shared, persistent store.

Usage:
    # Deploy the server and get its URL
    flyte run examples/apps/mlflow_server.py --deploy

    # Then pass that URL to your training pipeline, e.g.:
    flyte run examples/ml/model_selection_mlflow.py model_selection_pipeline \
        --mlflow_tracking_uri "https://<your-mlflow-app-endpoint>"
"""

import pathlib

import flyte
import flyte.app

image = flyte.Image.from_uv_script(__file__, name="mlflow-server")

mlflow_env = flyte.app.AppEnvironment(
    name="mlflow-server",
    image=image,
    # MLflow's built-in server command — serves both the API and the UI.
    # --host 0.0.0.0 binds to all interfaces so the Flyte ingress can reach it.
    # --backend-store-uri uses SQLite on a local path; swap to a managed DB
    # (e.g. postgresql://...) for production durability.
    args=[
        "mlflow",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--default-artifact-root",
        "/tmp/mlflow-artifacts",
        "--no-serve-artifacts",
        "--workers",
        "1",
    ],
    # Disable MLflow's host-validation middleware — in this context Flyte's
    # ingress handles auth/TLS and the Host header won't match localhost.
    env_vars={"MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE": "true"},
    port=8080,
    resources=flyte.Resources(cpu=1, memory="4Gi"),
    requires_auth=False,
    links=[flyte.app.Link(path="/", title="MLflow UI", is_relative=True)],
    scaling=flyte.app.Scaling(
        replicas=(0, 1),  # Scale from 0 to 1 replica
        scaledown_after=1800,  # Scale down after 5 minutes of inactivity
    ),
)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.serve(mlflow_env)
    print(f"MLflow UI: {app.url}")
