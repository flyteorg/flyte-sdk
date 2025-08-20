import logging
import pathlib

import flyte
from src.my_module import say_hello

env = flyte.TaskEnvironment(
    name="workflow_env",
)


@env.task
async def greet(name: str) -> str:
    return await say_hello(name)


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent
    config_path = current_dir / "../../../../config.yaml"
    flyte.init_from_config(str(config_path), log_level=logging.DEBUG, root_dir=current_dir.parent)
    r = flyte.run(greet, name="World")
    print(r.url)
