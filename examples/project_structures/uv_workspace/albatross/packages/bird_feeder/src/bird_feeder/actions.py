from pathlib import Path

from seeds.utils import get_default_seed_name

import flyte

UV_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent.parent

bird_env = flyte.TaskEnvironment(
    name="bird_feeder",
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=(UV_WORKSPACE_ROOT / "pyproject.toml"),
        extra_args="packages/bird_feeder --no-install-project",
    ),
)


# @bird_env.task
def get_feeder():
    print("Get bird feeder")
    print(f"Default seed: {get_default_seed_name()}")
