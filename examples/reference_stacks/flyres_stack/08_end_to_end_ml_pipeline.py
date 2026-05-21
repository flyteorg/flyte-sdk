"""
End-to-End ML Pipeline: Data → Training → Serving

This example demonstrates a complete ML lifecycle using Flyte, showing how it can replace
Dagster for data engineering and SkyPilot for distributed training.
"""

# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte",
#     "ray[default]>=2.40.0",
#     "torch==2.4.0",
#     "transformers>=4.40.0",
# ]
# ///

import asyncio
import os
from datetime import datetime
from typing import Any, Dict

import ray
from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

import flyte
from flyte import Image, Resources


def get_base_image() -> Image:
    """Base image with core ML libraries."""
    return (
        Image.from_debian_base(name="ml-pipeline", python_version=(3, 12))
        .with_pip_packages("flyte", "torch==2.4.0", "transformers==4.42.0")
        .with_apt_packages("curl", "wget")
    )


def get_ray_image() -> Image:
    """Image with Ray for distributed training."""
    return Image.from_debian_base(name="ml-pipeline-ray", python_version=(3, 12)).with_pip_packages(
        "flyteplugins-ray",
        "ray[default]==2.46.0",
        "torch==2.4.0",
        "transformers==4.42.0",
    )


# Configuration
HF_MOUNT_PATH = os.environ.get("HF_MOUNT_PATH", "/models")
RAY_HEAD_PORT = 6379

# Ray cluster for distributed training
ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(ray_start_params={"log-color": "True"}),
    worker_node_config=[WorkerNodeConfig(group_name="training-group", replicas=2)],
    runtime_env={
        "pip": ["torch", "transformers"],
        "env_vars": {"HF_HOME": HF_MOUNT_PATH},
    },
    enable_autoscaling=False,
)


# Environments
base_env = flyte.TaskEnvironment(name="ml_base", image=get_base_image())
ray_env = flyte.TaskEnvironment(
    name="ml_ray",
    plugin_config=ray_config,
    image=get_ray_image(),
    resources=Resources(cpu=(4, 8), memory=("8Gi", "32Gi"), gpu="L4:1"),
)


@base_env.task
async def ingest_data() -> Dict[str, Any]:
    """
    Step 1: Data ingestion.

    Simulates ingesting and preprocessing data for ML training.
    """
    await asyncio.sleep(0.5)  # Simulate I/O

    return {
        "data_source": "production_dataset",
        "records_processed": 10000,
        "features_extracted": ["text", "metadata", "labels"],
        "ingestion_timestamp": datetime.utcnow().isoformat(),
    }


@base_env.task
async def validate_data(data_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Data validation.

    Validates data quality before training.
    """
    await asyncio.sleep(0.3)  # Simulate validation

    return {
        **data_info,
        "validation_status": "passed",
        "quality_metrics": {
            "completeness": 0.98,
            "consistency": 0.95,
            "validity": 0.99,
        },
    }


@base_env.task
async def prepare_features(data_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: Feature engineering.

    Prepares features for model training.
    """
    await asyncio.sleep(0.4)  # Simulate feature extraction

    return {
        **data_info,
        "features": ["text_embedding", "metadata_features"],
        "feature_dimensions": {"embedding": 768, "metadata": 10},
    }


@ray_env.task
async def train_model(
    features: Dict[str, Any],
    epochs: int = 3,
) -> Dict[str, Any]:
    """
    Step 4: Distributed model training.

    Uses Ray for distributed training (replaces SkyPilot's async execution).
    """
    if not ray.is_initialized():
        ray.init(address="auto")

    @ray.remote
    def train_worker(worker_id: int) -> dict:
        return {
            "worker_id": worker_id,
            "loss_history": [1.0 - i * 0.1 for i in range(epochs)],
        }

    # Run distributed workers
    futures = [train_worker.remote(i) for i in range(3)]
    results = ray.get(futures)

    return {
        "training_status": "complete",
        "epochs_trained": epochs,
        "workers_used": 3,
        "final_metrics": {
            "train_loss": 0.7,
            "val_loss": 0.85,
            "accuracy": 0.82,
        },
    }


@base_env.task
async def evaluate_model(training_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 5: Model evaluation.

    Evaluates model performance on validation set.
    """
    metrics = training_result["final_metrics"]

    return {
        **training_result,
        "evaluation_status": "success",
        "metrics_summary": {
            "accuracy": metrics.get("accuracy", 0.82),
            "precision": 0.79,
            "recall": 0.81,
            "f1_score": 0.80,
        },
    }


@base_env.task
async def register_model(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 6: Model registration.

    Registers model in model registry for serving.
    """
    return {
        **evaluation,
        "model_registry": {
            "registry_name": "flyte-ml-models",
            "model_name": "text-classifier-v1",
            "version": "1.0.0",
            "status": "registered",
        },
        "deployment_ready": evaluation.get("metrics_summary", {}).get("accuracy", 0) > 0.7,
    }


@base_env.task
async def deploy_model(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 7: Model deployment.

    Deploys model for inference serving.
    """
    if not model_info.get("deployment_ready"):
        return {
            **model_info,
            "deployment_status": "not_ready",
            "reason": "Model accuracy below threshold",
        }

    await asyncio.sleep(0.3)  # Simulate deployment

    return {
        **model_info,
        "deployment_status": "active",
        "serving_endpoint": "/api/v1/models/text-classifier-v1/predict",
        "replicas": 2,
        "resource_allocation": {
            "cpu": "2 cores",
            "memory": "4Gi",
            "gpu": "None (CPU serving)",
        },
    }


@base_env.task
async def monitor_model_performance(deployment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 8: Model monitoring.

    Continuously monitors model performance in production.
    """
    return {
        **deployment,
        "monitoring_status": "active",
        "metrics_tracked": [
            "latency_p50",
            "latency_p95",
            "error_rate",
            "prediction_count",
        ],
        "alerts_configured": True,
    }


@base_env.task(triggers=flyte.Trigger.hourly())
async def scheduled_retraining(trigger_time: datetime) -> Dict[str, Any]:
    """
    Step 9: Scheduled retraining.

    Replaces Dagster's schedule triggers with Flyte cron triggers for periodic training.
    """
    print(f"Starting scheduled retraining at {trigger_time}")

    # Run the full pipeline
    data_info = await ingest_data()
    validated = await validate_data(data_info)
    features = await prepare_features(validated)
    training = await train_model(features, epochs=2)
    evaluation = await evaluate_model(training)
    registered = await register_model(evaluation)
    deployment = await deploy_model(registered)

    return {
        "scheduled_retraining_complete": True,
        "triggered_at": trigger_time.isoformat(),
        "final_deployment": deployment,
    }


async def run_full_pipeline() -> Dict[str, Any]:
    """Run the complete end-to-end ML pipeline."""
    flyte.init_from_config()

    print("=" * 60)
    print("End-to-End ML Pipeline (Data → Training → Serving)")
    print("=" * 60)

    # Step 1: Data ingestion
    print("\n[Step 1] Ingesting data...")
    data_info = flyte.run(ingest_data)
    print(f"Data ingested: {data_info.outputs[0]}")

    # Step 2: Validate data
    print("\n[Step 2] Validating data...")
    validated = flyte.run(validate_data, data_info=data_info.outputs[0])
    print(f"Validation status: {validated.outputs[0]['validation_status']}")

    # Step 3: Prepare features
    print("\n[Step 3] Preparing features...")
    features = flyte.run(prepare_features, data_info=validated.outputs[0])
    print(f"Features prepared: {features.outputs[0]['features']}")

    # Step 4: Train model (distributed)
    print("\n[Step 4] Training model (distributed)...")
    training = flyte.run(train_model, features=features.outputs[0], epochs=3)
    print(f"Training complete: {training.outputs[0]}")

    # Step 5: Evaluate
    print("\n[Step 5] Evaluating model...")
    evaluation = flyte.run(evaluate_model, training_result=training.outputs[0])
    print(f"Evaluation metrics: {evaluation.outputs[0]['metrics_summary']}")

    # Step 6: Register model
    print("\n[Step 6] Registering model...")
    registered = await flyte.run(register_model, evaluation=evaluation.outputs[0])
    print(f"Model registered: {registered.outputs[0]['model_registry']['model_name']}")

    # Step 7: Deploy
    print("\n[Step 7] Deploying model...")
    deployment = await flyte.run(deploy_model, model_info=registered.outputs[0])
    print(f"Deployment status: {deployment.outputs[0]['deployment_status']}")

    # Step 8: Monitor
    print("\n[Step 8] Monitoring model...")
    monitoring = await flyte.run(monitor_model_performance, deployment=deployment.outputs[0])
    print(f"Monitoring active: {monitoring.outputs[0]['metrics_tracked']}")

    return monitoring.outputs[0]


if __name__ == "__main__":
    import asyncio

    # Run the pipeline
    result = asyncio.run(run_full_pipeline())
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
