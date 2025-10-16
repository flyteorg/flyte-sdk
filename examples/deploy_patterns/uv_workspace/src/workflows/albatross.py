from pathlib import Path

from bird_feeder.actions import get_feeder
from seeds.actions import get_seed
from seeds.constants import hello, env_cons, albatross_task2

import flyte

env = flyte.TaskEnvironment(
    name="albatross_env",
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=Path("./pyproject.toml"),
        extra_args="--only-group all",
    ),
    depends_on=[env_cons]
)


@env.task
async def albatross_task() -> str:
    get_feeder()
    hello()
    await albatross_task2()
    print("123345678", flush=True)
    seed = get_seed(seed_name="Sun Flower seed")
    return f"Get bird feeder and feed with {seed}"


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    flyte.init_from_config(root_dir=current_dir.parent)
    run = flyte.run(albatross_task)
    print(run.url)
