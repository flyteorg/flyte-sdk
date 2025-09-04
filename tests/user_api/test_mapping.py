import dataclasses
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pytest
from pydantic import BaseModel, Field

import flyte
from flyte.io import Dir, File



def test_map_partials():
    from functools import partial

    # Define an environment with specific resources
    env = flyte.TaskEnvironment(name="map-test", resources=flyte.Resources(cpu="1"))

    @env.task
    def my_task(batch_id: int, name: str, constant_param: str) -> str:
        print(name, constant_param, batch_id)
        return name

    @env.task
    def main() -> List[str]:
        compounds = list(range(100))
        constant_param = "shared_config"

        curry_consts = partial(my_task, constant_param=constant_param, name='daniel')

        return list(
            flyte.map(
                curry_consts,
                compounds
            )
        )

    flyte.init()
    run = flyte.with_runcontext(mode="local").run(main)
    print(run.outputs())
