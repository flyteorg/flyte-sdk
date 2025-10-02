# Dynamic Environment Configuration Based on Domain

This pattern demonstrates how to configure different execution environments (dev, staging, production) based on the Flyte domain using `flyte.current_domain()`.

## The Pattern

Use `flyte.current_domain()` to deterministically create different task environments based on the deployment domain:

```python
def create_env():
    if flyte.current_domain() == "development":
        return flyte.TaskEnvironment(name="dev", image=..., env_vars={"MY_ENV": "dev"})
    return flyte.TaskEnvironment(name="prod", image=..., env_vars={"MY_ENV": "prod"})

env = create_env()

@env.task
async def my_task(n: int) -> int:
    return n + 1
```

## Why This Pattern?

**Environment reproducibility in local and remote clusters is critical.** Flyte re-instantiates modules in remote clusters, so `current_domain()` will be set correctly based on where the code executes.

L **Don't use environment variables directly** - they won't yield correct results unless you manually pass them to the downstream system.

 **Do use `flyte.current_domain()`** - Flyte automatically sets this based on the execution context.

## Important Constraints

`flyte.current_domain()` only works **after** `flyte.init()` is called:

-  Works with `flyte run` and `flyte deploy` CLI commands (they init automatically)
-  Works when called from `if __name__ == "__main__"` after explicit `flyte.init()`
- L Does NOT work at module level without initialization

## Usage

### With CLI (Recommended)
```bash
flyte run environment_picker.py entrypoint --n 5
flyte deploy environment_picker.py
```

### Programmatically
See `main.py` for how to properly initialize before importing:

```python
import flyte.git

flyte.init_from_config(flyte.git.config_from_root())
from environment_picker import entrypoint  # Import after init
```

## How It Works

1. Flyte sets the domain context when initializing
2. `current_domain()` returns the domain string (e.g., "development", "staging", "production")
3. Your code deterministically configures resources based on this domain
4. When Flyte executes remotely, it re-instantiates modules with the correct domain context
5. The same environment configuration logic runs consistently everywhere