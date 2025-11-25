"""
This is the only file that needs all your dependencies to be installed, all other files are independent.
So this can serve to be a local entrypoint, where you can use uv to install everything locally.
"""

from pathlib import Path

from root_env import env
from spark_env import env as spark_env
from torch_env import env as torch_env

import flyte

env.add_dependency(torch_env, spark_env)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    v = flyte.deploy(env)
    print(v[0].summary_repr())

    from root_env import root_task

    r = flyte.run(root_task)
    print(r.url)
