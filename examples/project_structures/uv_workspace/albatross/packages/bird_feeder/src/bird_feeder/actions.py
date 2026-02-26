from pathlib import Path

from seeds.utils import get_default_seed_name

import flyte

UV_PROJECT_ROOT = Path(__file__).parent.parent.parent  # bird_feeder

bird_env = flyte.TaskEnvironment(
    name="bird_feeder",
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=(UV_PROJECT_ROOT / "pyproject.toml"),
    ),
)


@bird_env.task
def get_feeder():
    print("Get bird feeder")
    print(f"Default seed: {get_default_seed_name()}")


if __name__ == '__main__':
    flyte.init_from_config(root_dir=UV_PROJECT_ROOT.parent)
    run = flyte.run(get_feeder)
    print(run.url)
