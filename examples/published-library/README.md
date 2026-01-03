# Using Flyte Tasks from Published Libraries

This example demonstrates how to use Flyte tasks that are imported from published Python packages on PyPI.

## Overview

When your Flyte tasks are packaged and published as a library (e.g., to PyPI), they can be reused across multiple projects. This example shows the proper configuration needed for Flyte to resolve and execute these tasks at runtime.

## Two Flavors of Usage

### Flavor 1: Local Task Using Library Tasks

If you have a local task that calls library tasks, use normal `flyte.run()`:

```python
@env.task
async def use_library(data: str = "default string", n: int = 3) -> str:
    result = await library_task(data, n)
    return result + " from consumer"

# Run normally - code bundle will be created
run = flyte.run(use_library, data="hello world", n=10)
```

This approach bundles your local code but still uses the library tasks from `site-packages`.

### Flavor 2: Running Library Tasks Directly

For efficiency, run library tasks directly without copying local code using `copy_style="none"`:

```python
# Run library task directly - no code bundling
run = flyte.with_runcontext(
    copy_style="none",  # Skip code bundling for efficiency
    version="0.4.0",    # Required when copy_style="none"
).run(library_task, data="hello world", n=5)
```

**Key Points:**
- `copy_style="none"` tells Flyte to skip code bundling entirely
- A version is **required** when using `copy_style="none"`
- Best practice: Use the library's version (e.g., `version=my_task_library.__version__`)
- The task is invoked directly from the package installed in `site-packages`

## Key Requirements

### Install the Library in the Image

The published library must be installed in the container image using `with_pip_packages()`:

```python
library_environment = flyte.TaskEnvironment(
    name="my-task-library-env",
    image=flyte.Image.from_debian_base().with_pip_packages("my-task-library"),
)
```

**Best Practice**: Use versioned packages to ensure consistency:

```python
.with_pip_packages("my-task-library==1.0.0")
```

## How It Works

1. **Development**: The task library (`my-task-library`) defines tasks with an environment that installs itself
2. **Publishing**: The library is published to PyPI
3. **Installation**: Consumer projects install the library via `uv pip install my-task-library`
4. **Runtime**:
   - **Flavor 1** (local task calling library tasks): Local code is bundled, library tasks run from `site-packages`
   - **Flavor 2** (direct library task execution): No code bundling, tasks invoked directly from `site-packages`
   - The image builder ensures the package is installed in `site-packages`
   - When using `copy_style="none"`, Flyte uses the version to track deployments

## Example Structure

- **`my-task-library/`**: The published package containing Flyte tasks
  - Defines tasks in `flyte_entities.py`
  - Specifies its own environment with `with_pip_packages("my-task-library")`

- **`my-flyte-project/`**: Consumer project that uses the library
  - Imports tasks: `from my_task_library import flyte_entities`
  - **Flavor 1**: Defines `use_library` task that calls library tasks, runs with normal `flyte.run()`
  - **Flavor 2**: Runs library tasks directly with `copy_style="none"`

## Important Limitations

- Only packages installed in `site-packages` or `dist-packages` are supported
- Other installation types (editable installs, custom paths) are **not** supported with `copy_style="none"`
- The package must be available at the exact version specified in both the image and the version parameter

## Installation

To install the task library locally for development:

```bash
uv pip install my-task-library
```

Or install a specific version:

```bash
uv pip install my-task-library==0.4.0
```
