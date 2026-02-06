"""Repro: nested Pydantic+FlyteDataFrame deadlock via Python API.

    PYTHONPATH=. .venv/bin/python test_pydantic_flyte_df_pyapi.py
"""

import signal
import sys
import threading
import traceback

import flyte
import pandas as pd
from flyte.io import DataFrame as FlyteDataFrame
from pydantic import BaseModel

env = flyte.TaskEnvironment(
    name="repro",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("pandas", "pyarrow"),
)


class MyModel(BaseModel):
    """Pydantic model wrapping a FlyteDataFrame."""

    data: FlyteDataFrame
    model_config = {"arbitrary_types_allowed": True}


@env.task
async def inner(model: MyModel) -> str:
    """Receives nested FlyteDataFrame."""
    return "OK"


@env.task
async def inner_bare(data: FlyteDataFrame) -> str:
    """Receives bare FlyteDataFrame."""
    return "OK"


def _timeout_handler(signum: int, frame: object) -> None:
    print("\n=== TIMEOUT (120s) - DEADLOCK ===")
    for tid, stack in sys._current_frames().items():
        name = next(
            (t.name for t in threading.enumerate() if t.ident == tid),
            "unknown",
        )
        print(f"\n--- {name} ({tid}) ---")
        traceback.print_stack(stack)
    sys.exit(1)


if __name__ == "__main__":
    import logging

    nested = "--bare" not in sys.argv

    flyte.init_from_config(log_level=logging.DEBUG)

    df = pd.DataFrame({"a": [1, 2, 3]})
    print("syncing df")
    fdf = FlyteDataFrame.from_local_sync(df)
    print("df synced")

    if nested:
        print("Calling flyte.run() with NESTED Pydantic model...")
        run = flyte.run(inner, model=MyModel(data=fdf))
    else:
        print("Calling flyte.run() with BARE FlyteDataFrame...")
        run = flyte.run(inner_bare, data=fdf)

    signal.alarm(0)
    print(f"SUCCESS: {run}")
