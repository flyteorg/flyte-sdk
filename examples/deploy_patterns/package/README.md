# Package Structure Example

This example demonstrates how to organize Flyte 2.0 workflows in a package structure with shared task environments and utilities.

## Structure

```
lib
├── __init__.py
├── workflows
│   ├── __init__.py
│   ├── workflow1.py
│   ├── workflow2.py
│   ├── env.py
│   ├── utils.py
```

The task environment is defined in `env.py` and shared across the two workflows. The workflows import from the shared environment and utilities:

```python
from lib.workflows.env import env
from lib.workflows import utils
```

## Running Locally

When running workflows locally, you need to update the `PYTHONPATH` environment variable:

```bash
PYTHONPATH=.:$PYTHONPATH flyte run lib/workflows/workflow1.py process_workflow
```

Or for workflow2:

```bash
PYTHONPATH=.:$PYTHONPATH flyte run lib/workflows/workflow2.py math_workflow --n 6
```

### Why PYTHONPATH is Required

When Flyte loads modules locally, it assumes it's running in a correctly configured Python environment and does **not** automatically modify the Python path. This is intentional behavior because:

1. **Avoiding Unintended Side Effects**: Automatically adding the current directory to `PYTHONPATH` could lead to:
   - Module name conflicts with system or site packages
   - Accidental shadowing of standard library modules
   - Inconsistent behavior between local development and remote execution
   - Difficulties in reproducing bugs if the local environment differs from production

2. **Environment Isolation**: Modifying `PYTHONPATH` globally could affect other Python processes and tools running on your system.

3. **Explicit Configuration**: Requiring explicit `PYTHONPATH` configuration makes dependencies clear and prevents "it works on my machine" issues.

### Runtime Behavior

At runtime, when code is deployed to a Flyte cluster:
- Flyte packages and copies your code to the execution environment
- The current working directory is automatically added to `PYTHONPATH` in the runtime container
- This ensures your package imports work correctly in the remote environment

The local behavior intentionally differs to prevent assumptions that might not hold in production.

## Alternative: Using a Python Project

A common and recommended solution is to create a proper Python project with a `pyproject.toml` file. This allows you to install your package in development mode:

```toml
# pyproject.toml
[project]
name = "lib"
version = "0.1.0"

[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"
```

Then install in editable mode:

```bash
pip install -e .
```

After this, you can run workflows without modifying `PYTHONPATH`:

```bash
flyte run lib/workflows/workflow1.py process_workflow
```

This approach is cleaner for larger projects and provides better integration with Python tooling.
