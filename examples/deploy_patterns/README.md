# UV Project Deployment Patterns

Modern Python dependency management and deployment patterns using UV with Flyte. This guide covers three main patterns for different use cases and shows you exactly when and how to use each one.

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
image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=pathlib.Path("pyproject.toml"),
    pre=True,
    # Install only the dependencies specified in pyproject.toml, without installing the current project itself.
    # Only pyproject.toml and uv.lock will be uploaded to the remote builder to resolve and install dependencies.
    extra_args="--no-install-project"
)
```

**pyproject.toml:**
```toml
[project]
name = "my-flyte-app"
dependencies = [
    "flyte>=2.0.0",
    "requests",
    "numpy",
]

[project.optional-dependencies]
ml = ["pandas", "scikit-learn"]
```

### 2. uv_project_lib: Custom Libraries

**Project Structure:**
```
uv_project_lib/
├── main.py                        # Alternative main application file
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
image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=pathlib.Path("my_plugin/pyproject.toml"),  # Point to library's pyproject.toml
    pre=True,
    extra_args="--inexact"  # TODO: Set it as default in the SDK
)
```

**Library Structure:**
```python
# my_plugin/pyproject.toml
[project]
name = "my-custom-lib"
dependencies = ["numpy"]

# main.py
from my_lib.math_utils import linear_function
```

### 3. uv_workspace: Monorepo

**Project Structure:**
```
uv_workspace/
├── pyproject.toml      # Workspace definition
├── uv.lock             # Workspace lock file
├── main.py             # Alternative main application file
├── src/
│   └── tasks/          # Main application tasks
│       └── main.py
├── bird_feeder/        # Package 1
│   ├── pyproject.toml
│   └── src/bird_feeder/
│       └── actions.py
└── seeds/              # Package 2
    ├── pyproject.toml
    └── src/seeds/
        └── actions.py
```

**When to use:**
- Multiple related packages in one repository
- Shared dependencies across packages
- Complex project structures

**Key Configuration:**
```python
# From tasks/main.py - points to workspace root
image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=(UV_WORKSPACE_ROOT / "pyproject.toml"),  # Root workspace pyproject.toml
    extra_args="--only-group albatross --inexact"  # Install specific dependency groups
)

# From main.py - local pyproject.toml reference
image = flyte.Image.from_debian_base().with_uv_project(
    pyproject_file=Path(__file__).parent / "pyproject.toml",  # Install all dependencies in the workspace
    extra_args="--inexact"
)
```

**Workspace Structure:**
```toml
# Root pyproject.toml
[tool.uv.workspace]
members = ["bird_feeder", "seeds"]

[tool.uv.sources]
bird-feeder = { workspace = true }
seeds = { workspace = true }

[dependency-groups]
albatross = [
    "bird-feeder",
    "seeds",
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

**Trade-offs:**
- 📡 **Network Dependency**: Requires code bundle download at runtime
- 🔄 **Less Reproducible**: Code changes without image versioning

### Production (Full Build)
```python
import flyte
import pathlib

env = flyte.TaskEnvironment(
    name="full_build",
    image=(
        flyte.Image.from_debian_base()
        .with_uv_project(pyproject_file=pathlib.Path("my_plugin/pyproject.toml"))
        .with_source_file("./main.py")
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
    "seeds",
    "numpy"
]

# Data processing task dependencies
data_processing = [
    "bird-feeder",
    "seeds", 
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

