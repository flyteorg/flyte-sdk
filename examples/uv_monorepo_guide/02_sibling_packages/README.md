# 02_sibling_packages

Two independent packages in the same repo: `my_lib` (a pure Python utility library) and `my_app` (Flyte tasks). Each has its own `pyproject.toml` and `uv.lock`.

## What this example shows

- `my_app/pyproject.toml` declares `my-lib` as an editable path dep so the image builder can find its source during `uv sync`
- `with_source_folder(MY_LIB_PKG)` bakes `my_lib` source into the image at `/root/my_lib/`, making it importable at runtime
- `root_dir = my_app/src/` covers only `my_app`; `my_lib` does not need to be in the code bundle
- `my_app/uv.lock` is the deployment lockfile used for image builds
- The root `pyproject.toml` and `uv.lock` are dev-only conveniences

## Setup

Install both packages as editable for local development (uses the root `pyproject.toml`):

```bash
cd 02_sibling_packages
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

Uncomment the `flyte.deploy(...)` line and comment out `flyte.run(...)` in `my_app/src/my_app/main.py`, then:

```bash
uv run python my_app/src/my_app/main.py
```

## Project structure

```
02_sibling_packages/
+-- pyproject.toml        <- dev-only: installs both packages as editable
+-- uv.lock               <- dev-only lockfile
+-- my_lib/
|   +-- pyproject.toml   <- standalone lib, no Flyte dep
|   +-- src/my_lib/
|       +-- __init__.py
|       +-- stats.py     <- mean, variance, std
+-- my_app/
    +-- pyproject.toml   <- lists flyte + my-lib as editable path dep
    +-- uv.lock          <- deployment lockfile (run `cd my_app && uv lock`)
    +-- src/my_app/
        +-- __init__.py
        +-- env.py       <- TaskEnvironment with with_source_folder(my_lib)
        +-- tasks.py     <- task definitions
        +-- main.py      <- entry point, workflow definition
```