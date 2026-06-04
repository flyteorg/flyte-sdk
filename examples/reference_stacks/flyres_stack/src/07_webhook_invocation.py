"""
Webhook Invocation: External Workflow Triggers

This example demonstrates Flyte's webhook app for external workflow invocation,
which replaces Dagster's external job triggers and SkyPilot's async execution model.
"""

# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte",
# ]
# ///

from datetime import datetime
from typing import Any, Dict

import flyte
from flyte import Image


def get_webhook_image() -> Image:
    """Get image for webhook-triggered tasks."""
    return Image.from_debian_base(name="webhooks", python_version=(3, 12)).with_pip_packages("flyte")


image = get_webhook_image()
base_env = flyte.TaskEnvironment(
    name="webhook_tasks",
    image=image,
)


@base_env.task
async def webhook_triggered_task(payload: Dict[str, Any], metadata: Dict[str, str]) -> Dict[str, Any]:
    """
    Task triggered via webhook.

    Replaces:
    - Dagster's external pipeline triggers
    - SkyPilot's async job launch via `sky launch`

    Webhook app env: https://www.union.ai/docs/v2/union/user-guide/task-deployment/invoke-webhook/
    """
    return {
        "task": "webhook_triggered_task",
        "payload": payload,
        "metadata": metadata,
        "triggered_at": datetime.utcnow().isoformat(),
        "status": "processed",
    }


@base_env.task
async def trigger_pipeline_run(
    pipeline_name: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Trigger a Flyte workflow run in response to external events.

    This replaces SkyPilot's async job submission + Flyte's pipeline orchestration.
    """
    return {
        "pipeline_name": pipeline_name,
        "parameters": parameters,
        "triggered_at": datetime.utcnow().isoformat(),
        "status": "triggered",
    }


@base_env.task
async def process_webhook_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an incoming webhook event.

    This is the main entry point for external systems to trigger Flyte workflows.
    """
    # Route based on event type
    event_type = event.get("type", "unknown")

    if event_type == "model_ready":
        return await handle_model_ready_event(event)
    elif event_type == "data_update":
        return await handle_data_update_event(event)
    elif event_type == "training_complete":
        return await handle_training_complete_event(event)
    else:
        return {
            "status": "unknown_event_type",
            "event_type": event_type,
        }


async def handle_model_ready_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle model ready events."""
    return await trigger_pipeline_run(
        pipeline_name="model_serving_pipeline",
        parameters={
            "model_path": event.get("model_path"),
            "version": event.get("version"),
        },
    )


async def handle_data_update_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle data update events."""
    return await trigger_pipeline_run(
        pipeline_name="data_retraining_pipeline",
        parameters={
            "dataset_version": event.get("dataset_version"),
            "retrain_model": True,
        },
    )


async def handle_training_complete_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle training completion events."""
    return await trigger_pipeline_run(
        pipeline_name="evaluation_pipeline",
        parameters={
            "model_checkpoint": event.get("checkpoint"),
            "metrics": event.get("metrics", {}),
        },
    )


# Example of a webhook-activated workflow
@base_env.task
async def end_to_end_workflow(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete ML workflow triggered by webhook.

    Replaces SkyPilot's async job chain with Flyte's orchestration.
    """

    # Step 1: Data ingestion (triggered by webhook)
    data_info = await trigger_pipeline_run(
        pipeline_name="data_ingestion",
        parameters={"source": payload.get("data_source", "default")},
    )

    # Step 2: Model training
    train_result = await trigger_pipeline_run(
        pipeline_name="model_training",
        parameters={
            "dataset_version": data_info.get("parameters", {}).get("version"),
            "epochs": 10,
        },
    )

    # Step 3: Evaluation
    eval_result = await webhook_triggered_task(
        payload={"stage": "evaluation"},
        metadata=train_result,
    )

    return {
        "workflow_complete": True,
        "steps_completed": ["data_ingestion", "model_training", "evaluation"],
        "final_result": eval_result,
    }


async def simulate_external_trigger() -> None:
    """
    Simulate an external system triggering Flyte workflows.

    In production, this would be:
    - A CI/CD pipeline
    - An ML monitoring system
    - A SkyPilot job completion callback
    - Any external HTTP webhook
    """
    flyte.init_from_config()

    # Simulate webhook payload from external system
    event = {
        "type": "model_ready",
        "model_path": "/models/trained_model_v1",
        "version": "v1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }

    print(f"Simulating webhook event: {event}")

    result = await process_webhook_event(event)
    print(f"Webhook processing result: {result.outputs[0]}")


if __name__ == "__main__":
    flyte.init_from_config()

    print("=" * 60)
    print("Flyte Webhook Invocation (SkyPilot Alternative)")
    print("=" * 60)

    # Simulate webhook event
    print("\n[Webhook] Processing incoming event...")
    result = flyte.run(
        process_webhook_event,
        event={
            "type": "data_update",
            "dataset_version": "v2.0.0",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
    print(f"Result: {result.outputs[0]}")

    # Show how to trigger end-to-end workflows
    print("\n[Workflow] Triggering complete ML workflow...")
    workflow_result = flyte.run(
        end_to_end_workflow,
        payload={
            "data_source": "production_dataset",
            "triggered_by": "webhook",
        },
    )
    print(f"Workflow result: {workflow_result.outputs[0]}")
