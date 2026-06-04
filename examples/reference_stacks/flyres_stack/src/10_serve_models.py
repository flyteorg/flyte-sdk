"""
Model Serving: Online and Batch Inference

This example demonstrates Flyte's model serving capabilities, including online endpoints
and batch inference jobs - alternatives to SkyPilot's serving infrastructure.
"""

# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte",
# ]
# ///

import asyncio
from typing import Any, Dict, List, Optional

import flyte
from flyte import Image, Resources


def get_serve_image() -> Image:
    """Get image with serving runtime."""
    return Image.from_debian_base(name="model-serve", python_version=(3, 12)).with_pip_packages(
        "flyte",
        "torch==2.4.0",
        "transformers==4.42.0",
        "fastapi>=0.100.0",
    )


image = get_serve_image()
base_env = flyte.TaskEnvironment(name="serve_base", image=image)
serving_env = flyte.TaskEnvironment(
    name="model_serving",
    image=image,
    resources=Resources(cpu=(2, 4), memory=("1Gi", "4Gi")),
)


class ModelEndpoint:
    """Represents a deployed model endpoint."""

    def __init__(
        self,
        model_name: str,
        version: str,
        endpoint_url: str,
        replicas: int = 1,
        resource_allocation: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.version = version
        self.endpoint_url = endpoint_url
        self.replicas = replicas
        self.resource_allocation = resource_allocation or {
            "cpu": "1 core",
            "memory": "2Gi",
        }
        self.created_at = asyncio.get_event_loop().time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "endpoint_url": self.endpoint_url,
            "replicas": self.replicas,
            "resource_allocation": self.resource_allocation,
            "created_at": asyncio.get_event_loop().time(),
        }


@base_env.task
async def load_model_for_serving(
    model_path: str,
) -> Dict[str, Any]:
    """
    Load a model for serving.

    Pre-loads models into memory for low-latency inference.
    """
    await asyncio.sleep(0.1)  # Simulate loading

    return {
        "model_loaded": True,
        "model_path": model_path,
        "load_time_ms": 150,  # Simulated
    }


@base_env.task
async def deploy_model_endpoint(
    model_info: Dict[str, Any],
    endpoint_name: str,
    replicas: int = 2,
) -> ModelEndpoint:
    """
    Deploy a model as an online inference endpoint.

    Replaces SkyPilot's service deployment with Flyte task orchestration.
    """
    # In production, this would:
    # 1. Spin up serving pods
    # 2. Configure load balancer
    # 3. Set up health checks

    endpoint = ModelEndpoint(
        model_name=endpoint_name,
        version="v1",
        endpoint_url=f"/api/v1/models/{endpoint_name}/predict",
        replicas=replicas,
        resource_allocation={
            "cpu": f"{replicas * 0.5} cores total",
            "memory": f"{replicas * 2}Gi total",
        },
    )

    return endpoint


@base_env.task
async def health_check(endpoint: ModelEndpoint) -> Dict[str, Any]:
    """
    Check health of deployed model endpoint.
    """
    await asyncio.sleep(0.1)  # Simulate request

    return {
        "endpoint": endpoint.endpoint_url,
        "status": "healthy",
        "replicas_healthy": endpoint.replicas,
        "latency_p50_ms": 12,
        "latency_p95_ms": 45,
    }


@base_env.task
async def predict(
    endpoint: ModelEndpoint,
    input_data: List[str],
) -> List[Dict[str, Any]]:
    """
    Run inference on a deployed model.

    Simulates online prediction (in production, would call actual endpoint).
    """
    # In production, this would:
    # 1. Send request to serving endpoint
    # 2. Parse responses

    results = []
    for i, text in enumerate(input_data):
        results.append(
            {
                "input": text,
                "prediction": f"class_{i % 3}",
                "confidence": round(0.7 + (i * 0.05), 2),
                "endpoint": endpoint.endpoint_url,
            }
        )

    return results


@base_env.task
async def batch_inference(
    model_info: Dict[str, Any],
    input_paths: List[str],
) -> List[Dict[str, Any]]:
    """
    Run batch inference on multiple inputs.

    Replaces SkyPilot's batch job with Flyte orchestration.
    """
    # In production, this would:
    # 1. Load all models
    # 2. Process batches in parallel
    # 3. Save results

    results = []
    for path in input_paths:
        await asyncio.sleep(0.1)  # Simulate processing

        results.append(
            {
                "input_path": path,
                "output_path": f"{path}.predictions",
                "records_processed": 100,  # Simulated
                "inference_time_ms": 500,  # Simulated
            }
        )

    return results


@base_env.task
async def monitor_endpoint(
    endpoint: ModelEndpoint,
) -> Dict[str, Any]:
    """
    Monitor endpoint performance and health.
    """
    return {
        "endpoint": endpoint.endpoint_url,
        "metrics": {
            "requests_per_second": 150.5,
            "error_rate": 0.001,
            "p50_latency_ms": 12,
            "p95_latency_ms": 45,
            "p99_latency_ms": 89,
        },
        "status": "healthy",
    }


async def run_serving_example():
    """Run the model serving example."""
    flyte.init_from_config()

    print("=" * 60)
    print("Model Serving: Online and Batch Inference")
    print("=" * 60)

    # Step 1: Load model
    print("\n[Load] Loading model for serving...")
    model_info = await flyte.run(
        load_model_for_serving,
        model_path="/models/final_checkpoint.pth",
    )
    print(f"Model loaded: {model_info.outputs[0]}")

    # Step 2: Deploy endpoint
    print("\n[Deploy] Deploying model endpoint...")
    endpoint = await flyte.run(
        deploy_model_endpoint,
        model_info=model_info.outputs[0],
        endpoint_name="text-classifier",
        replicas=3,
    )
    print(f"Endpoint deployed: {endpoint.outputs[0].to_dict()}")

    # Step 3: Health check
    print("\n[Health] Checking endpoint health...")
    health = await flyte.run(health_check, endpoint=endpoint.outputs[0])
    print(f"Health status: {health.outputs[0]}")

    # Step 4: Online inference
    print("\n[Predict] Running online inference...")
    predictions = await flyte.run(
        predict,
        endpoint=endpoint.outputs[0],
        input_data=["Hello world", "Test sentence", "Another example"],
    )
    print(f"Predictions: {predictions.outputs[0]}")

    # Step 5: Batch inference
    print("\n[Batch] Running batch inference...")
    batch_results = await flyte.run(
        batch_inference,
        model_info=model_info.outputs[0],
        input_paths=[
            "/data/batch_1.parquet",
            "/data/batch_2.parquet",
        ],
    )
    print(f"Batch results: {batch_results.outputs[0]}")

    # Step 6: Monitor
    print("\n[Monitor] Monitoring endpoint...")
    monitoring = await flyte.run(monitor_endpoint, endpoint=endpoint.outputs[0])
    print(f"Monitoring: {monitoring.outputs[0]}")


if __name__ == "__main__":
    import asyncio

    # asyncio.run(run_serving_example())

    print("Model Serving Example")
    print("=====================")
    print()
    print("Features:")
    print("- Online endpoints with auto-scaling replicas")
    print("- Health checks and monitoring")
    print("- Batch inference jobs")
    print("- Performance metrics tracking")
