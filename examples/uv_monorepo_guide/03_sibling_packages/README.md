# 03_sibling_packages

Two independent packages in the same repo: `my_lib` (a pure Python utility library) and `my_app` (Flyte tasks). Each has its own `pyproject.toml`. `my_app/uv.lock` is the deployment lockfile used for image builds.

## What this example shows

- `my_app/pyproject.toml` lists only external PyPI deps -- `my_lib` is not declared as a dependency there
- `my_lib` source arrives via the code bundle at runtime; it is not installed in the image
- The root `pyproject.toml` is a dev convenience that installs both packages as editable for local development
- `root_dir` is the repo root, covering both `my_app/src/` and `my_lib/src/`
- `my_app/uv.lock` is the deployment lockfile; the root `uv.lock` is for local dev only

## Setup

Install both packages as editable for local development (uses the root `pyproject.toml`):

```bash
cd 03_sibling_packages
uv sync
```

Generate the deployment lockfile for image builds:

```bash
cd my_app && uv lock && cd ..
```

## Run (fast deploy -- development)

```bash
uv run python my_app/src/my_app/main.py
```

## Run (full build -- production)

To bake source into the image, change `flyte.run(...)` in `my_app/src/my_app/main.py` to:

```python
flyte.deploy(env, copy_style="none", version="1.0.0")
```

## Project structure

```
03_sibling_packages/
+-- pyproject.toml        <- dev-only: installs both packages as editable
+-- uv.lock               <- dev-only lockfile
+-- my_lib/
|   +-- pyproject.toml   <- standalone lib, no Flyte dep
|   +-- src/my_lib/
|       +-- __init__.py
|       +-- stats.py     <- mean, variance, std
+-- my_app/
    +-- pyproject.toml   <- external PyPI deps only (flyte); used for image builds
    +-- uv.lock          <- deployment lockfile (run `cd my_app && uv lock`)
    +-- src/my_app/
        +-- __init__.py
        +-- tasks.py     <- TaskEnvironment + task definitions
        +-- main.py      <- entry point, workflow definition
```