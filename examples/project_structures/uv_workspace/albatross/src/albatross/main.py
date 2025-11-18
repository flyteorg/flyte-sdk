from pathlib import Path

from albatross.condor.strategy import get_strategy
from bird_feeder.actions import bird_env, get_feeder
from seeds.actions import get_seed

import flyte

UV_WORKSPACE_ROOT = Path(__file__).parent.parent.parent

env = flyte.TaskEnvironment(
    name="uv_workspace",
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=(UV_WORKSPACE_ROOT / "pyproject.toml"),
        extra_args="--only-group albatross",  # albatross group define all the dependencies the task needs
    ),
    depends_on=[bird_env],
)


@env.task
async def albatross_task() -> str:
    get_feeder()
    get_strategy()
    seed = get_seed(seed_name="Sun Flower seed")
    return f"Get bird feeder and feed with {seed}"


if __name__ == "__main__":
    flyte.init_from_config(root_dir=UV_WORKSPACE_ROOT)
    run = flyte.run(albatross_task)
    print(run.url)
