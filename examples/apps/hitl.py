"""
Human-in-the-Loop (HITL) Pattern Example

This example demonstrates how to implement a human-in-the-loop workflow using
Flyte tasks and the flyteplugins-hitl plugin. The pattern allows a workflow to
pause and wait for human input before continuing execution.

Architecture:
1. The hitl plugin provides an Event class that encapsulates a FastAPI app for human input
2. When an Event is created, it automatically serves the app using flyte.serve
3. The workflow orchestrates automated tasks with HITL checkpoints

The HITL functionality uses an event-based API:
- `hitl.new_event(...)` creates an event and starts the FastAPI app
- `event.wait()` or `await event.wait.aio()` blocks until input is received
- Supports different data types (int, float, str, bool)
- Supports different scopes ("run" for run-level events)

Usage:
    python examples/apps/hitl.py

The workflow will:
1. Run task1() which returns an integer
2. Create an Event (which starts the app) and wait for human input
3. Print a URL where the human can submit their input
4. Once input is received, continue to task2() with both values

Example event-based API usage:
    # Create an event (starts the app) and wait for input
    event = await hitl.new_event.aio(
        "my_event",
        data_type=int,
        scope="run",
        prompt="Enter a number",
    )
    value = await event.wait.aio()
"""

import logging

import flyte
import flyteplugins.hitl as hitl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Task Environment and Tasks
# ============================================================================

task_env = flyte.TaskEnvironment(
    name="hitl-workflow",
    image=(
        flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn", "python-multipart")
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[hitl.env],
)


@task_env.task(report=True)
async def task1() -> int:
    """
    First task in the workflow - returns an automated value.
    """
    logger.info("task1: Computing automated value...")
    result = 42
    logger.info(f"task1: Returning {result}")
    await flyte.report.replace.aio(f"task1: Returning {result}")
    await flyte.report.flush.aio()
    return result


@task_env.task
async def task2(x: int, y: int) -> int:
    """
    Second task that combines automated and human input.
    """
    logger.info(f"task2: Received x={x}, y={y}")
    result = x + y
    logger.info(f"task2: Returning {result}")
    return result


@task_env.task(report=True)
async def main() -> int:
    """
    Main workflow that orchestrates automated and human-in-the-loop tasks.

    Flow:
    1. task1() runs and returns an automated value (x)
    2. Create an Event (serves the app) and wait for human input (y)
    3. task2(x, y) combines both values and returns the result
    """
    print("Starting HITL workflow...")

    # Step 1: Automated task
    x = await task1()
    print(f"task1 completed: x = {x}")

    # Step 2: Human-in-the-loop using the Event-based API
    # Create an event (this serves the app if not already running)
    event = await hitl.new_event.aio(
        "integer_input_event",
        data_type=int,
        scope="run",
        prompt="What should I add to x?",
    )
    y = await event.wait.aio()
    print(f"Event completed: y = {y}")

    # Step 3: Combine results
    result = await task2(x, y)
    print(f"task2 completed: result = {result}")

    return result


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HITL Example")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in local mode (for testing)",
    )
    args = parser.parse_args()

    flyte.init_from_config(log_level=logging.DEBUG)

    print("\nStarting HITL workflow...")
    run = flyte.run(main)
    print(f"Run URL: {run.url}")
    print(f"Run name: {run.name}")

    print("\nWaiting for workflow to complete...")
    print("(Remember to submit human input when prompted!)")
    run.wait()

    outputs = run.outputs()
    print(f"\nWorkflow completed! Result: {outputs}")
