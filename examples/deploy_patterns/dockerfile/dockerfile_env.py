"""
Usage:

Run:
```
flyte run dockerfile_env.py main --x 10
```

Deploy:
```
flyte --config ../../../config.yaml deploy dockerfile_env.py env
```
"""

from pathlib import Path

import flyte

env = flyte.TaskEnvironment(
    name="docker_env",
    image=flyte.Image.from_dockerfile(
        # relative paths in python change based on where you call, so set it relative to this file
        Path(__file__).parent / "Dockerfile",
        registry="ghcr.io/flyteorg",
        name="docker_env_image",
    ),
)


@env.task
def main(x: int) -> int:
    return x * 2


if __name__ == "__main__":
    from pathlib import Path

    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())

    run = flyte.run(main, x=10)
    print(run.url)
