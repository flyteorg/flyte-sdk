# Sibling Packages
This example differs from the uv workspace example in that each package can be deployed separately and each come with its own `uv.lock` file. The base `uv.lock` file is only there because there is also a base `pyproject.toml` file.

## Running from CLI

```bash
cd uv_project_local_dep/my_app
uv sync && . .venv/bin/activate
cd ..  # back to uv_project_local_dep

flyte -c <config> run --root-dir . my_app/src/my_app/main.py process_list --x_list '[1,2,3]' 
```
