# Full Build Deployment Pattern

This example demonstrates how to configure Flyte to **completely copy your code into the container** and disable Flyte's default fast deployment system.

## Overview

By default, Flyte uses a fast deployment system that:
- Creates a tar archive of your files
- Skips the full image build and push process
- Provides faster iteration during development

However, sometimes you need to **completely embed your code into the container image**. This is useful when:
- You want full reproducibility with immutable container images
- You need to deploy to environments where the fast deployment system isn't available
- You want to ensure all dependencies are baked into the image
- You're preparing for production deployments

## Key Configuration

To achieve full code copying into containers, you need to configure three main components:

### 1. Set `copy_style` to `"none"`

```python
flyte.with_runcontext(copy_style="none", version="x").run(main, n=10)
```

This disables Flyte's fast deployment system and forces a full container build.

### 2. Set a Custom Version

```python
flyte.with_runcontext(copy_style="none", version="x").run(main, n=10)
```

The `version` parameter should be set to a desired value (not auto-generated) to ensure consistent image tagging.

### 3. Configure Image Source Copying

```python
image=flyte.Image.from_debian_base().with_source_folder(
    pathlib.Path(__file__).parent,
    copy_contents_only=True
)
```

Use `.with_source_folder()` to specify what code to copy into the container.

### 4. Set `root_dir` Correctly

```python
flyte.init_from_config(
    flyte.git.config_from_root(),
    root_dir=pathlib.Path(__file__).parent
)
```

## Configuration Options

### Image Source Copying Options

#### Option A: Copy Folder Structure
```python
# Copies the entire folder structure into the container
image=flyte.Image.from_debian_base().with_source_folder(
    pathlib.Path(__file__).parent,
    copy_contents_only=False  # Default
)

# When copy_contents_only=False, set root_dir to parent.parent
flyte.init_from_config(
    flyte.git.config_from_root(),
    root_dir=pathlib.Path(__file__).parent.parent
)
```

#### Option B: Copy Contents Only (Recommended)
```python
# Copies only the contents of the folder (flattens structure)
image=flyte.Image.from_debian_base().with_source_folder(
    pathlib.Path(__file__).parent,
    copy_contents_only=True
)

# When copy_contents_only=True, set root_dir to parent
flyte.init_from_config(
    flyte.git.config_from_root(),
    root_dir=pathlib.Path(__file__).parent
)
```

## Complete Example

```python
import pathlib
import flyte
from dep import foo

# Configure task environment with source copying
env = flyte.TaskEnvironment(
    name="full_build",
    image=flyte.Image.from_debian_base().with_source_folder(
        pathlib.Path(__file__).parent,
        copy_contents_only=True
    ),
)

@env.task
def square(x) -> int:
    return x ** foo()  # Uses local dependency

@env.task
def main(n: int) -> list[int]:
    return list(flyte.map(square, range(n)))

if __name__ == "__main__":
    import flyte.git

    # Initialize with correct root_dir for copy_contents_only=True
    flyte.init_from_config(
        flyte.git.config_from_root(),
        root_dir=pathlib.Path(__file__).parent
    )

    # Run with full build (no fast deployment)
    run = flyte.with_runcontext(
        copy_style="none",  # Disable fast deployment
        version="v1.0"      # Set explicit version
    ).run(main, n=10)

    print(run.url)
```

## Important Notes

### Root Directory Configuration

The `root_dir` setting must match your `copy_contents_only` choice:

- **`copy_contents_only=True`**: Set `root_dir=pathlib.Path(__file__).parent`
- **`copy_contents_only=False`**: Set `root_dir=pathlib.Path(__file__).parent.parent`

This ensures Flyte can correctly resolve relative imports and file paths.

### Version Management

When using `copy_style="none"`, always specify an explicit version:
- Use semantic versioning: `"v1.0.0"`, `"v1.1.0"`
- Use build numbers: `"build-123"`
- Use git commits: `"abc123"`

Avoid auto-generated versions to ensure reproducible deployments.

### Performance Considerations

- **Full builds take longer** than fast deployment
- **Container images will be larger** as they include all source code
- **Better for production** where immutability is important
- **Use during development** when you need to test the full deployment pipeline

## When to Use This Pattern

✅ **Use full build when:**
- Deploying to production environments
- Need immutable, reproducible container images
- Working with complex dependency structures
- Deploying to air-gapped or restricted environments
- Building CI/CD pipelines

❌ **Don't use full build when:**
- Rapid development and iteration
- Working with frequently changing code
- Development environments where speed matters
- Simple workflows without complex dependencies

## Troubleshooting

### Common Issues

1. **Import errors**: Check your `root_dir` configuration matches `copy_contents_only`
2. **Missing files**: Ensure all dependencies are in the source folder
3. **Version conflicts**: Use explicit, unique version strings
4. **Build failures**: Check that the base image has all required system dependencies

### Debug Tips

- Add print statements to verify file paths in containers
- Use `docker run -it <image> /bin/bash` to inspect built images
- Check Flyte logs for build errors and warnings
- Verify that relative imports work correctly in the container context