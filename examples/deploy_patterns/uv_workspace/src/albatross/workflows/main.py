from pathlib import Path

from bird_feeder.actions import get_feeder
from albatross.condor.strategy import get_strategy

import flyte

env = flyte.TaskEnvironment(
    name="albatross_env",
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=Path("./pyproject.toml"),
        extra_args="--only-group main",
    ),
)


@env.task
async def albatross_task() -> str:
    get_feeder()
    seed = "Sun Flower seed"
    condor = get_strategy()
    return f"Get bird feeder and feed with {seed}. {condor=}"


if __name__ == "__main__":
    print(f"main task in main.py {get_feeder()}", flush=True)
    # current_dir = Path(__file__).parent
    # flyte.init_from_config(root_dir=current_dir.parent)
    # run = flyte.run(albatross_task)
    # print(run.url)
