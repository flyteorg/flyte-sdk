"""
Workflow Triggers: Cron and Event-Based Execution

This example demonstrates Flyte's trigger system, which replaces Dagster's schedule
triggers with cron-based execution. Also shows how to use webhook invocations.
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


def get_trigger_image() -> Image:
    """Get image for trigger-based tasks."""
    return Image.from_debian_base(name="triggers", python_version=(3, 12)).with_pip_packages("flyte")


image = get_trigger_image()
base_env = flyte.TaskEnvironment(
    name="trigger_tasks",
    image=image,
)


# Example 1: Hourly task
@base_env.task(triggers=flyte.Trigger.hourly())
async def hourly_task(trigger_time: datetime) -> Dict[str, Any]:
    """
    Run every hour.

    Replaces Dagster's @schedule with Flyte's Trigger.hourly().
    """
    return {
        "run_type": "hourly",
        "triggered_at": trigger_time.isoformat(),
        "task_status": "completed",
    }


# Example 2: Daily task at specific time
@base_env.task(triggers=flyte.Trigger.daily())
async def daily_task(trigger_time: datetime) -> Dict[str, Any]:
    """
    Run every day at midnight.

    Replaces Dagster's @daily_schedule with Flyte's Trigger.daily().
    """
    return {
        "run_type": "daily",
        "triggered_at": trigger_time.isoformat(),
        "task_status": "completed",
    }


# Example 3: Custom cron schedule
custom_cron = flyte.Trigger(
    name="weekly_training_run",
    definition=flyte.Cron("0 2 * * 0", timezone="UTC"),  # Every Sunday at 2 AM UTC
    inputs={
        "run_name": str,
        "model_version": str,
        "dry_run": False,
    },
)


@base_env.task(triggers=(custom_cron,))
async def weekly_training_run(
    trigger_time: datetime,
    run_name: str,
    model_version: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Weekly training job triggered by cron.

    Replaces Dagster's custom schedule with inputs using Flyte triggers with inputs.
    """
    return {
        "run_type": "weekly",
        "triggered_at": trigger_time.isoformat(),
        "run_name": run_name,
        "model_version": model_version,
        "dry_run": dry_run,
        "status": "scheduled",
    }


# Example 4: Workflow with scheduled triggers
@base_env.task(triggers=flyte.Trigger.minutely())
async def monitoring_task(trigger_time: datetime) -> Dict[str, Any]:
    """
    Monitor system health every minute.

    Replaces Dagster's sensor-based monitoring with Flyte triggers.
    """
    return {
        "timestamp": trigger_time.isoformat(),
        "status": "healthy",
        "checks_passed": True,
    }


# Example 5: On-demand trigger (manual execution)
manual_trigger = flyte.Trigger(
    name="on_demand_analysis",
    definition=flyte.Manual(),
    inputs={
        "data_version": str,
        "analysis_type": str,
    },
)


@base_env.task(triggers=(manual_trigger,))
async def manual_data_analysis(
    data_version: str,
    analysis_type: str,
) -> Dict[str, Any]:
    """
    Manual trigger for on-demand analysis.

    Replaces Dagster's manual run execution with Flyte's manual triggers.
    """
    return {
        "data_version": data_version,
        "analysis_type": analysis_type,
        "triggered_manually": True,
    }


# Example 6: Conditional execution based on trigger
@base_env.task(triggers=flyte.Trigger.hourly())
async def conditional_task(trigger_time: datetime) -> Dict[str, Any]:
    """
    Task that behaves differently based on trigger context.

    Demonstrates how triggers can provide context for conditional logic.
    """
    is_weekend = trigger_time.weekday() >= 5

    return {
        "run_type": "conditional",
        "timestamp": trigger_time.isoformat(),
        "is_weekend": is_weekend,
        "actions_taken": [
            "check_data_quality" if not is_weekend else "skip_non_critical_checks",
            "generate_report",
        ],
    }


def deploy_triggers(env: flyte.TaskEnvironment) -> None:
    """
    Deploy triggers to Flyte backend.

    This replaces Dagster's schedule deployment with Flyte's trigger registration.
    """
    print(f"Deploying triggers for environment: {env.name}")
    # In production, use: flyte.deploy(env)


def register_triggers() -> None:
    """Register all triggers with the Flyte backend."""
    print("Available triggers:")
    print("- hourly_task: Runs every hour")
    print("- daily_task: Runs at midnight daily")
    print("- weekly_training_run: Runs Sunday at 2 AM UTC")
    print("- monitoring_task: Runs every minute")
    print("- manual_data_analysis: Triggered manually via CLI/API")


if __name__ == "__main__":
    flyte.init_from_config()

    print("=" * 60)
    print("Flyte Workflow Triggers (Dagster Schedule Alternative)")
    print("=" * 60)

    # Deploy the environment
    deploy_triggers(base_env)

    # Register all triggers
    register_triggers()

    # Run a manual example
    print("\n[Manual] Running on-demand analysis...")
    result = flyte.run(
        manual_data_analysis,
        data_version="v1.0",
        analysis_type="daily_summary",
    )
    print(f"Result: {result.outputs[0]}")
