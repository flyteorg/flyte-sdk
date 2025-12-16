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

## Running Locally or Remote

When running workflows with a package structure, use the `--root-dir` flag to specify the root directory of your package:

```bash
flyte run --root-dir . lib/workflows/workflow1.py process_workflow
```

Or for workflow2:

```bash
flyte run --root-dir . lib/workflows/workflow2.py math_workflow --n 6
```

### How `--root-dir` Works

The `--root-dir` flag automatically sets the Python path (`sys.path`) to the location pointed by `--root-dir`. This ensures that:

1. **Local execution**: Your package imports work correctly when running locally
2. **Consistent behavior**: The same Python path configuration is used both locally and at runtime
3. **No manual PYTHONPATH**: You don't need to manually export or modify `PYTHONPATH` environment variables

### Runtime and Deploy Behavior

When code is executed remotely (via `flyte run` without `--local` or `flyte deploy`):
- Flyte packages and copies your code to the execution environment
- The same package structure is preserved in the runtime container
- The `--root-dir` location is automatically added to `sys.path` in the runtime environment
- This ensures your package imports work identically in both local and remote execution

The `flyte deploy` command follows the same pattern:
```bash
flyte deploy --root-dir . lib/workflows/workflow1.py
```

This will package the code from the root directory and configure the runtime to use the same Python path.

## Alternative: Using a Python Project

For larger projects, you can create a proper Python project with a `pyproject.toml` file:

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

After this, you can run workflows without the `--root-dir` flag:

```bash
flyte run lib/workflows/workflow1.py process_workflow
```

However, when deploying or running remotely, you should still use `--root-dir` to ensure consistent package structure:

```bash
flyte run --root-dir . lib/workflows/workflow1.py process_workflow
flyte deploy --root-dir . lib/workflows/workflow1.py
```

This approach provides better integration with Python tooling and dependency management while maintaining consistency between local and remote execution.
