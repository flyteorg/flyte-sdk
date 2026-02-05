"""
Example: Accessing wandb run directories in both async and sync tasks.

This example demonstrates:
1. Using download_logs parameter in @wandb_init decorator
2. Async tasks with wandb logging
3. Sync tasks with wandb logging
4. Manual downloads using download_wandb_run_dir()
"""

import flyte
from flyte.io import Dir

from flyteplugins.wandb import (
    download_wandb_run_dir,
    get_wandb_run,
    get_wandb_run_dir,
    wandb_config,
    wandb_init,
)

env = flyte.TaskEnvironment(
    name="wandb-dir-example",
    image=flyte.Image.from_debian_base(name="wandb-dir-example").with_pip_packages(
        "flyteplugins-wandb"
    ),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


@wandb_init(download_logs=True, run_mode="new")
@env.task
async def async_child_with_download(x: int) -> str:
    """Async task that automatically downloads logs via download_logs=True."""
    run = get_wandb_run()

    # Log some metrics
    run.log({"async_metric": x * 2, "input_value": x})

    print(f"Async child task - Run ID: {run.id}")

    return run.id


@wandb_init(download_logs=False, run_mode="new")
@env.task
async def async_child_no_download() -> str:
    """Async task with new run that doesn't download logs."""
    run = get_wandb_run()

    run.log({"independent_async_metric": 42})
    print(f"Async child (new run) - Run ID: {run.id}")

    return run.id


@wandb_init(download_logs=True)
@env.task
def sync_child_with_download(value: int) -> str:
    """Sync task that automatically downloads logs via download_logs=True."""
    run = get_wandb_run()

    # Log some metrics
    run.log({"sync_metric": value * 3, "processed_value": value})

    # Get local run dir (available during task execution)
    local_dir = get_wandb_run_dir()
    print(f"Sync child task - Run ID: {run.id}, local dir: {local_dir}")

    # Write a custom file to the run directory
    with open(f"{local_dir}/sync_output.txt", "w") as f:
        f.write(f"Sync processing value: {value}")

    return run.id


@wandb_init(download_logs=False, run_mode="new")
@env.task
def sync_child_no_download(x: int) -> str:
    """Sync task with new run that doesn't download logs."""
    run = get_wandb_run()

    run.log({"another_sync_metric": x * 4})
    print(f"Sync child (new run) - Run ID: {run.id}")

    return run.id


@wandb_init(download_logs=True)
@env.task
async def parent_orchestrator() -> dict[str, str | Dir]:
    """Parent task that orchestrates children and manually downloads some logs."""

    print("Parent task - orchestrating async and sync child tasks")

    # Log some metrics
    run = get_wandb_run()
    run.log({"parent_metric": 100})

    # Call async children
    async_run_1 = await async_child_with_download(10)
    print(f"Async child 1 completed with run ID: {async_run_1}")

    async_run_2 = await async_child_no_download()
    print(f"Async child 2 completed with run ID: {async_run_2}")

    # Call sync children
    sync_run_1 = sync_child_with_download(25)
    print(f"Sync child 1 completed with run ID: {sync_run_1}")

    sync_run_2 = sync_child_no_download(30)
    print(f"Sync child 2 completed with run ID: {sync_run_2}")

    # Manually download logs for runs that didn't use download_logs=True
    # This is useful for parent tasks that want to collect all child logs
    print("\nManually downloading logs for async_run_2...")
    async_run_2_logs = download_wandb_run_dir(run_id=async_run_2)
    print(f"Downloaded to: {async_run_2_logs}")

    print("Manually downloading logs for sync_run_2...")
    sync_run_2_logs = download_wandb_run_dir(run_id=sync_run_2)
    print(f"Downloaded to: {sync_run_2_logs}")

    return {
        "async_run_1": async_run_1,
        "async_run_2": async_run_2,
        "async_run_2_logs": await Dir.from_local(async_run_2_logs),
        "sync_run_1": sync_run_1,
        "sync_run_2": sync_run_2,
        "sync_run_2_logs": await Dir.from_local(sync_run_2_logs),
    }


if __name__ == "__main__":
    flyte.init_from_config()

    print("Running Example: Wandb Directory Access (Async + Sync)")

    run = flyte.with_runcontext(
        custom_context=wandb_config(
            project="flyte-wandb-test",
            entity="samhita-alla",
            tags=["dir-example", "async-sync"],
            download_logs=False,  # Default for context, can be overridden per task
        ),
    ).run(parent_orchestrator)

    print(f"\nRun URL: {run.url}")
