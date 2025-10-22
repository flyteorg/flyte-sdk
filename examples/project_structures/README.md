# UV Project Deployment Patterns

Modern Python dependency management and deployment patterns using UV with Flyte. This guide covers three main patterns for different use cases and shows you exactly when and how to use each one.

## When to Use `with_uv_project`

The `with_uv_project` method is essential in three specific scenarios:

**Use Case 1: Installing Locked Dependencies**
When you want to install the exact dependencies specified in `uv.lock`:
```python
image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=pathlib.Path(__file__).parent / "pyproject.toml",
)
```
This ensures reproducible builds by using the locked versions from `uv.lock`, while `--no-install-project` prevents installing the current project itself (only dependencies).

**Use Case 2: Installing pyproject Without Code Copying**
When you want to install the python project in the image without copying it into the code bundle (using `copy_style=None`):
```python
image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=pathlib.Path(__file__).parent / "pyproject.toml",
    project_install_mode="install_project",
)

# Deploy with copy_style="none" to bake code into image
flyte.with_runcontext(copy_style="none", version="v1.0").run(task_function)
```
This is ideal for production deployments where you want immutable, self-contained container images with all dependencies pre-installed.

**Use Case 3: Multi-Path Projects (Plugin in Different Location)**
When your task's pyproject and a plugin's pyproject are in different paths (e.g., one in `/tmp`, another in `/root/uv_workspace/`). In this scenario, fast register never works, so you must use `with_uv_project` to install the plugin in your image:
```python
# Example: Plugin located outside the main project structure
image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=pathlib.Path("/tmp/my_plugin/pyproject.toml"),
    project_install_mode="install_project",
)
```
Fast register requires all code to be in a single tree for the code bundle, but when dependencies span multiple directories, you need `with_uv_project` to explicitly install them during the image build phase.


## 🚀 Quick Decision Tree

```
Do you want a single-file solution?
├─ YES → Use uvscript (inline dependencies, no pyproject.toml)
└─ NO → Do you have a multi-package workspace?
    ├─ YES → Use uv_workspace (monorepo with shared packages)
    └─ NO → Do you need custom libraries?
        ├─ YES → Use uv_project_lib (project with local packages)  
        └─ NO → Use uv_project (simple project dependencies)
```

## 📋 Pattern Overview

| Pattern | Use Case | Project Structure | Key Benefits |
|---------|----------|------------------|--------------|
| **uv_project** | Simple projects with external deps | Single pyproject.toml | Fast, simple dependency management |
| **uv_project_lib** | Projects with local packages | Single pyproject.toml for library | Custom library integration |
| **uv_workspace** | Monorepo/multi-package | UV workspace with members | Shared dependencies, package isolation |
| **uvscript** | Single-file scripts | No pyproject.toml | Inline dependencies, zero config files |

## 📚 Pattern Details & Examples

### 1. uv_project: Simple Dependencies

**Project Structure:**
```
uv_project/
├── pyproject.toml      # Project dependencies
├── uv.lock             # Lock file
└── main.py             # Main application
```

**When to use:**
- External libraries only (requests, numpy, pandas)
- Single application without custom packages
- Fast iteration and development

**Key Configuration:**
```python
env = flyte.TaskEnvironment(
    name="pyproject_test",
    resources=flyte.Resources(memory="250Mi"),
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=pathlib.Path(__file__).parent / "pyproject.toml",
    )
)
```

**pyproject.toml:**
```toml
[project]
name = "tester"
version = "0.1.0"
description = "Test project"
requires-python = ">=3.12"
dependencies = [
    "requests",
    "numpy",
]

[project.optional-dependencies]
pandas = ["pandas", "numpy"]
```

### 2. uv_project_lib: Custom Libraries

**Project Structure:**
```
uv_project_lib/
└── my_plugin/
    ├── pyproject.toml             # Library definition
    ├── uv.lock                    # Library lock file
    └── src/my_lib/
        ├── __init__.py
        ├── main.py                # Main application tasks
        └── math_utils.py          # Custom library code
```

**When to use:**
- Custom Python packages alongside external dependencies
- Local library development
- Code reusability across projects

**Key Configuration:**
```python
UV_PROJECT_ROOT = Path(Path(__file__)).parent.parent.parent

env = flyte.TaskEnvironment(
    name="uv_project_lib_task_in_src",
    resources=flyte.Resources(memory="1000Mi"),
    image=(
        flyte.Image.from_debian_base()
        .with_uv_project(
            pyproject_file=UV_WORKSPACE_ROOT / "pyproject.toml",
        )
    ),
)
```

### 3. uv_workspace: Monorepo

**Project Structure:**
```
uv_workspace/
└── albatross/
    ├── pyproject.toml          # Workspace definition
    ├── uv.lock                 # Workspace lock file
    ├── src/
    │   └── albatross/          # Main application
    │       ├── __init__.py
    │       ├── main.py
    │       └── condor/         # Application submodule
    │           ├── __init__.py
    │           └── strategy.py
    └── packages/
        ├── bird_feeder/        # Package 1
        │   ├── pyproject.toml
        │   └── src/bird_feeder/
        │       ├── __init__.py
        │       └── actions.py
        └── seeds/              # Package 2
            ├── pyproject.toml
            └── src/seeds/
                ├── __init__.py
                ├── actions.py
                └── utils.py
```

**When to use:**
- Multiple related packages in one repository
- Shared dependencies across packages
- Complex project structures

**Key Configuration:**

```python
# From src/albatross/main.py
from pathlib import Path

from bird_feeder.actions import get_feeder
from seeds.actions import get_seed
from albatross.condor.strategy import get_strategy

import flyte

UV_WORKSPACE_ROOT = Path(__file__).parent.parent.parent

env = flyte.TaskEnvironment(
    name="uv_workspace",
    image=flyte.Image.from_debian_base()
    .with_uv_project(
        pyproject_file=(UV_WORKSPACE_ROOT / "pyproject.toml"),
        extra_args="--only-group albatross",  # Install specific dependency group
        project_install_mode="install_project",
    ),
)


@env.task
async def albatross_task() -> str:
    get_feeder()
    get_strategy()
    seed = get_seed(seed_name="Sun Flower seed")
    return f"Get bird feeder and feed with {seed}"
```

**Workspace pyproject.toml:**
```toml
[project]
name = "albatross"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["bird-feeder"]

[tool.uv.sources]
bird-feeder = { workspace = true }
seeds = { workspace = true }

[tool.uv.workspace]
members = ["packages/*"]

[build-system]
requires = ["uv_build>=0.9.3,<0.10.0"]
build-backend = "uv_build"

[dependency-groups]
albatross = [
    "bird-feeder",
    "numpy"
]
```

## Development vs Production

### Development (Fast Iteration)
```python
import flyte
import pathlib

env = flyte.TaskEnvironment(
    name="fast_iteration",
    image=(
        flyte.Image.from_debian_base()
        .with_uv_project(pyproject_file=pathlib.Path("my_plugin/pyproject.toml"))
    ),
)

@env.task
def task_function() -> list[int]:
    ...

flyte.run(task_function)
```

**How it works under the hood:**
1. **Code Bundle Creation**: Flyte creates a compressed tar archive of your source code
2. **Lightweight Container**: Uses existing base image without rebuilding
3. **Runtime Code Injection**: Downloads and extracts the code bundle at container startup
4. **Fast Deployment**: Skips Docker build/push cycle, reducing deployment time from minutes to seconds

**Benefits:**
- ⚡ **Speed**: Faster than full builds for iteration and development
- 🔄 **Iteration**: Perfect for rapid development cycles
- 📦 **Small Images**: Base image stays unchanged

### Production (Full Build)
```python
import flyte
import pathlib

env = flyte.TaskEnvironment(
    name="full_build",
    image=(
        flyte.Image.from_debian_base()
        .with_uv_project(pyproject_file=pathlib.Path("my_plugin/pyproject.toml"), project_install_mode="install_project")
    ),
)

@env.task
def task_function() -> list[int]:
    ...

flyte.with_runcontext(copy_style="none", version="v1.0").run(task_function)
```

**How it works under the hood:**
1. **Complete Rebuild**: Triggers full Docker image build process
2. **Code Embedding**: All source code is baked into the container image layers
3. **Immutable Artifact**: Creates a self-contained, versioned container
4. **Registry Push**: Pushes the complete image to container registry
5. **No Runtime Dependencies**: Container runs independently without external code loading

**Benefits:**
- 🏗️ **Immutable**: Full reproducibility with versioned containers
- 🔒 **Self-Contained**: No runtime dependencies on code bundles
- 🚀 **Production Ready**: Suitable for air-gapped or restricted environments
- 📋 **Audit Trail**: Complete image history and versioning

**Trade-offs:**
- 💾 **Larger Images**: Contains all source code in image layers
- 🔄 **Less Flexible**: Requires rebuild for any code changes

### Dependency Group Strategy

**For workspaces, organize dependencies by tasks:**
```toml
[dependency-groups]
# Main albatross task dependencies (matches current example)
albatross = [
    "bird-feeder",
    "numpy"
]

# Data processing task dependencies
data_processing = [
    "bird-feeder",
    "pandas",
    "numpy"
]

# ML training task dependencies  
ml_training = [
    "bird-feeder",
    "scikit-learn",
    "torch"
]
```

## 🎯 Choosing the Right Pattern

### Decision Matrix

| Scenario | Recommended Pattern | Key Reason |
|----------|-------------------|------------|
| Simple ML pipeline with pandas/numpy | `uv_project` | External deps only |
| Reusable data processing library | `uv_project_lib` | Custom package sharing |
| Multiple related Python packages | `uv_workspace` | Package interdependencies |
| Quick prototype script | `uvscript` | Zero configuration |
| Production ML platform | `uv_workspace` + `copy_style="none"` | Scalability + immutability |

### Migration Path

1. **Start with `uv_project`** for proof of concept
2. **Move to `uv_project_lib`** when you need custom utilities
3. **Upgrade to `uv_workspace`** when managing multiple pyprojects
4. **Add full build** (`copy_style="none"`) for production deployment

## 🚀 Getting Started

1. **Choose your pattern** using the decision tree above
2. **Copy the relevant example** from this repository
3. **Customize the pyproject.toml** with your dependencies
4. **Run `uv lock`** to generate lock files
5. **Test locally** with `flyte run`
6. **Deploy** with confidence!

