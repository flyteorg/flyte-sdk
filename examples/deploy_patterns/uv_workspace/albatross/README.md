
# Flyte SDK: UV Workspace Deployment Pattern

UV workspace is useful for projects that have multiple project folders where some of them are dependencies of a main project. This example demonstrates how to set up a UV workspace in a Flyte task environment. In this example, the `bird_feeder` and `seeds` packages are workspace dependencies required by the Flyte task and will be installed in the UV workspace under editable mode.

## Project Structure

```
albatross/
├── pyproject.toml              # Workspace root configuration
├── uv.lock                     # Lock file for dependencies
├── packages/                   # Workspace packages
│   ├── bird_feeder/            # bird_feeder package
│   │   ├── pyproject.toml
│   │   └── src/bird_feeder/
│   │       ├── __init__.py
│   │       └── actions.py
│   └── seeds/                  # seeds package
│       ├── pyproject.toml
│       └── src/seeds/
│           ├── __init__.py
│           ├── actions.py
│           └── utils.py
└── src/                        # Main application source
    └── albatross/
        ├── __init__.py
        ├── main.py             # Flyte task definition
        └── condor/             # Application submodule
            ├── __init__.py
            └── strategy.py
```

##  Testing This Example
Execute the flyte command directly:
```bash
flyte -vvv -c <YOUR_CONFIG_FILE_PATH> run --root-dir `pwd`/src src/albatross/main.py albatross_task
```
Or just run python script:
```bash
python src/albatross/main.py
```

##  Issues With `--root-dir`
Flyte-SDK has helper functions that attempt to automatically detect the root folder. In uv workspace environments, the root directory might be
accidentally pointed to a dependency project directory instead of the main project. In this case, you need to explicitly specify the --root-dir
argument to override the automatic detected root directory.

##  Issues With `uv sync`
When running the `uv sync` command, uv will only install dependencies specified in the `dependencies` attribute in pyproject.toml, excluding all unspecified dependencies. Therefore, make sure to specify `flyte` in the dependencies section, or you may encounter a file not found error at runtime.

## Issues With `--only-group` Argument Of `uv sync` Command
It is recommended to specify dependency groups in order to exclude unrelated dependencies from being installed. You can follow the example below:

In pyproject.toml:
```toml
[project]
name = "albatross"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["bird-feeder", "seeds"]

[tool.uv.sources]
bird-feeder = { workspace = true }
seeds = { workspace = true }

[tool.uv.workspace]
members = ["packages/*"]

[dependency-groups]
albatross = [
    "bird-feeder",
    "seeds",
    "numpy"
]
```

In Flyte task environment:
```python
from pathlib import Path
import flyte

UV_WORKSPACE_ROOT = Path(__file__).parent.parent.parent

env = flyte.TaskEnvironment(
    name="uv_workspace",
    image=flyte.Image.from_debian_base()
    .with_uv_project(
        pyproject_file=(UV_WORKSPACE_ROOT / "pyproject.toml"),
        extra_args="--only-group albatross",  # Specify dependency group with --only-group argument
        project_install_mode="install_as_package",
    ),
)
```

## Best Practices

1. **Always set `root_dir`**: Ensure the root folder is not incorrectly detected by Flyte-SDK's auto-detection mechanism
2. **Always specify `flyte` in pyproject.toml dependencies**: Ensure flyte is not excluded by `uv sync`
3. **Always specify `dependency-groups` in pyproject.toml**: Ensure unrelated dependencies are not installed

## Common Pitfalls

1. **Forgetting to set `root_dir` path**: Results in path errors when registering a run
2. **Forgetting to specify `flyte` in pyproject.toml dependencies**: Results in import errors during remote execution