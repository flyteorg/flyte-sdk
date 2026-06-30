import pathlib

import flyte
from my_app.env import env
from my_app.tasks import compute_stats, summarize

MY_APP_ROOT = pathlib.Path(__file__).parent.parent.parent  # -> my_app/
SRC_DIR = MY_APP_ROOT / "src"  # -> my_app/src/


@env.task
async def stats_pipeline(values: list[float]) -> str:
    stats = await compute_stats(values=values)
    return await summarize(stats=stats)


if __name__ == "__main__":
    # my_lib is installed in the image; root_dir only needs to cover my_app source
    flyte.init_from_config(root_dir=SRC_DIR)

    # Development -- run a task directly, code bundle handles source delivery
    run = flyte.run(stats_pipeline, values=[1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Run URL: {run.url}")

    # Production -- deploy an environment with source baked into the image
    # flyte.deploy(env, copy_style="none", version="1.0.0")
