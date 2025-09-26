from pathlib import Path

from bird_feeder.actions import get_feeder
from seeds.actions import get_seed

import flyte

env = flyte.TaskEnvironment(
    name="albatross_env",
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=Path("./pyproject.toml"),
        extra_args="--only-group all",
    ),
)


@env.task
async def albatross_task() -> str:
    get_feeder()
    seed = get_seed(seed_name="Sun Flower seed")
    return f"Get bird feeder and feed with {seed}"


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    flyte.init_from_config(root_dir=current_dir.parent)
    run = flyte.run(albatross_task)
    print(run.url)
