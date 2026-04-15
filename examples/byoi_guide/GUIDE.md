# Bring Your Own Image in Flyte v2

This guide is for teams who own their Docker images and want Flyte for
orchestration without handing over their build pipeline.

> **Not covered here:** `flyte.Image.from_debian_base()`. That is the
> Flyte-managed image builder. This guide assumes you already have images.

---

## The Multi-Team Problem

Two teams. Two images. One workflow.

| | Team A (data-prep) | Team B (training) |
|---|---|---|
| Base | `python:3.11-slim` | `python:3.10-slim` (prod: CUDA) |
| Python | 3.11 | 3.10 |
| WORKDIR | `/app` | `/workspace` |
| PYTHONPATH | `/app` | `/workspace` |
| Packages | pandas, pyarrow | torch, numpy |

The `prepare` task runs in Team A's container. It processes the input and
calls `train`, which runs in Team B's container. One workflow, two images,
different filesystem layouts.

Two patterns solve this. Pick based on who controls what:

| | Pattern 1: Pure BYOI | Pattern 2: Remote Builder |
|---|---|---|
| Who owns the image? | Each team owns everything | Each team owns the base |
| Flyte-aware? | Yes — code is baked in | No — Flyte adapts on top |
| Code change = image rebuild? | Yes | No |
| Use when | Teams can't let Flyte touch images | Teams can hand off a base |

---

## Pattern 1: Pure BYOI

Teams build complete, Flyte-aware images. Workflow code is copied into the
Dockerfile via `COPY`. Flyte runs the container as a black box — it sends no code and
modifies nothing.

### Dockerfiles

Both teams install `flyte` and COPY the shared `workflow_code/` into their
image. The only difference is their base, Python version, and WORKDIR.

**data_prep/Dockerfile** (Team A):

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir flyte pandas==2.1.4 pyarrow==14.0.1

# Bake workflow code in. Every code change requires a new image tag + rebuild.
COPY workflow_code/ /app/workflow_code/

ENV PYTHONPATH=/app
```

**training/Dockerfile** (Team B):

```dockerfile
FROM python:3.10-slim   # prod: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir flyte torch==2.1.2 numpy==1.26.4

# /workspace/workflow_code/ — same import path as Team A despite different WORKDIR.
COPY workflow_code/ /workspace/workflow_code/

ENV PYTHONPATH=/workspace
```

Both expose `import workflow_code.tasks` at runtime because each image's
PYTHONPATH points to its own WORKDIR where the code was copied.

### Build and push

The build context is `pure_byoi/` so that `workflow_code/` is available to
both Dockerfiles:

```bash
# From v2_guide/pure_byoi/
docker build -f data_prep/Dockerfile -t <your-registry>/data-prep:latest .
docker build -f training/Dockerfile  -t <your-registry>/training:latest  .
docker push <your-registry>/data-prep:latest
docker push <your-registry>/training:latest
```

### Python code

**workflow_code/envs.py** — image names are specified via `from_ref_name()`:

```python
env_train = flyte.TaskEnvironment(
    name="training",
    image=flyte.Image.from_ref_name("training"),
)

env_data = flyte.TaskEnvironment(
    name="data-prep",
    image=flyte.Image.from_ref_name("data-prep"),
    depends_on=[env_train],
)
```

`from_ref_name()` is a placeholder resolved at runtime. The actual URIs are
passed in `main.py` via `init_from_config(images=...)`. This is necessary
because `envs.py` is copied into both images — hardcoding a URI would create
a circular reference where an image contains a reference to its own tag.

**main.py** — this is where image URIs are wired in:

```python
flyte.init_from_config(
    images=(
        "data-prep=<your-registry>/data-prep:latest",
        "training=<your-registry>/training:latest",
    ),
)
```

See [`pure_byoi/`](pure_byoi/) for the full example.

### Run and deploy

```bash
# From v2_guide/pure_byoi/
uv run main.py
```

There is no separate deploy step. The image tag is the version. To ship a
code change: edit tasks, rebuild both images, push new tags, update the tag
constants in `main.py`, run again.

---

## Pattern 2: Remote Builder

Teams hand you their base images. They built these images for their own
purposes — Flyte was never a consideration. Your job is to adapt them.

### The base images

**Team A** uses `continuumio/miniconda3` as their base. Their image has:
- Python at `/opt/conda/bin/python` (conda manages this)
- conda's own Dockerfile already adds `/opt/conda/bin` to `PATH`, so `python`
  is findable without any changes
- No PYTHONPATH set
- WORKDIR `/app`

**Team B** uses `python:3.10-slim` with a pip venv at `/opt/venv`. Their image has:
- Python at `/opt/venv/bin/python`
- `PATH` does **not** include `/opt/venv/bin` — the venv was created but never
  activated in the Dockerfile. Without activation, `python` on PATH resolves to
  the slim base's system Python, which has none of Team B's packages.
- No PYTHONPATH set
- WORKDIR `/workspace`

**data_prep/Dockerfile** (Team A's base — no Flyte, no workflow code):

```dockerfile
FROM continuumio/miniconda3:latest

RUN conda install -y -c conda-forge pandas==2.2.3 pyarrow==19.0.1 && \
    conda clean -afy

WORKDIR /app
```

**training/Dockerfile** (Team B's base — no Flyte, no workflow code):

```dockerfile
FROM python:3.10-slim   # prod: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir torch==2.1.2+cpu numpy==1.26.4

WORKDIR /workspace
# PATH does not include /opt/venv/bin — venv is not activated
```

### Build and push

Teams build base images infrequently — only when system deps change:

```bash
# From v2_guide/remote_builder/
docker build -f data_prep/Dockerfile -t <your-registry>/data-prep-base:latest data_prep/
docker build -f training/Dockerfile  -t <your-registry>/training-base:latest  training/
docker push <your-registry>/data-prep-base:latest
docker push <your-registry>/training-base:latest
```

### Adapting with `flyte.Image`

`flyte.Image.from_base()` takes the base URI and lets you layer on top of it.
This is where the image names are specified — one `.from_base()` call per team.

**Team A** only needs `flyte` installed and `PYTHONPATH` set. conda's PATH is
already correct:

```python
env_data_image = (
    flyte.Image.from_base("<your-registry>/data-prep-base:latest")
    .clone(name="<your-repo>", registry="<your-registry>", extendable=True)
    .with_commands(["/opt/conda/bin/pip install flyte"])
    .with_env_vars({"PYTHONPATH": "/app"})
    .with_code_bundle()
)
```

**Team B** needs three things: `flyte` installed in the venv, `PATH` updated
so the venv's `python` is the default, and `PYTHONPATH` set. `$PATH` in an
`ENV` instruction expands at Docker build time to the base image's PATH value:

```python
env_train_image = (
    flyte.Image.from_base("<your-registry>/training-base:latest")
    .clone(name="<your-repo>", registry="<your-registry>", extendable=True)
    .with_commands(["/opt/venv/bin/pip install flyte"])
    .with_env_vars({
        "PATH": "/opt/venv/bin:$PATH",   # activate the venv
        "PYTHONPATH": "/workspace",
    })
    .with_code_bundle()
)
```

`.with_code_bundle()` tells Flyte to inject task source at runtime (dev) or
bake it into the image at deploy time (prod). The code bundle extracts into
the WORKDIR — `/app` for Team A, `/workspace` for Team B — and each team's
`PYTHONPATH` makes it importable.

See [`remote_builder/`](remote_builder/) for the full example.

### Run and deploy

```bash
# From v2_guide/remote_builder/

# Development: fast code iteration — image only rebuilds when base image changes
uv run main.py

# Production: bake code into both images (uncomment in main.py)
# flyte.deploy(env_data, copy_style="none", version="1.0.0")
# flyte.deploy(env_train, copy_style="none", version="1.0.0")
```

During development you only rebuild the base image when the Dockerfile
changes. Code changes are free — they travel as a tarball at runtime.

---

## Decision Matrix

| Scenario | Pattern |
|---|---|
| Teams own full images, can't let Flyte touch them | Pure BYOI |
| Teams hand off a base image (no Flyte knowledge required) | Remote Builder |
| Code change should not require image rebuild | Remote Builder + `with_code_bundle()` |
| Base has non-standard Python location | `.with_commands()` to fix PATH before Flyte uses it |
| Production deploy, self-contained containers | `copy_style="none"` in `flyte.deploy()` |
