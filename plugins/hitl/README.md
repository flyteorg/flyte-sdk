# Flyte HITL Plugin

Human-in-the-Loop (HITL) plugin for Flyte. This plugin provides an event-based API for pausing workflows and waiting for human input.

## Installation

```bash
pip install flyteplugins-hitl
```

## Usage

```python
import flyte
import flyteplugins.hitl as hitl

task_env = flyte.TaskEnvironment(
    name="my-hitl-workflow",
    image=flyte.Image.from_debian_base(python_version=(3, 12)),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[hitl.env],
)


@task_env.task
async def task1() -> int:
    """First task - returns an automated value."""
    return 42


@task_env.task
async def task2(x: int, y: int) -> int:
    """Second task - combines automated and human input."""
    return x + y


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


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(f"Run URL: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")
```

## Features

- **Event-based API**: Create events that pause workflow execution until human input is received
- **Web Form**: Automatically generates a web form for human input
- **Programmatic API**: Submit input via curl or any HTTP client
- **Type Safety**: Supports int, float, str, and bool data types
- **Crash Resilient**: Uses durable sleep and object storage for state persistence

## API Reference

### `hitl.new_event(name, data_type, scope, prompt, ...)`

Create a new human-in-the-loop event.

**Parameters:**
- `name` (str): A descriptive name for the event
- `data_type` (Type): The expected type of the input (int, float, str, bool)
- `scope` (str): The scope of the event. Currently only "run" is supported.
- `prompt` (str): The prompt to display to the human
- `timeout_seconds` (int): Maximum time to wait for human input (default: 3600)
- `poll_interval_seconds` (int): How often to check for a response (default: 5)

**Returns:** An `Event` object

### `event.wait()` / `await event.wait.aio()`

Wait for human input and return the result.

**Returns:** The value provided by the human, converted to the event's data_type

### `hitl.env`

The `TaskEnvironment` that provides the HITL app infrastructure. Add this to your task environment's `depends_on` list.

### `hitl.Event`

The Event class for type annotations and advanced usage.
