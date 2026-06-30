"""Repro: nested Pydantic+FlyteDataFrame deadlock via Python API.

PYTHONPATH=. .venv/bin/python test_pydantic_flyte_df_pyapi.py
"""

import signal
import sys
from dataclasses import dataclass
from typing import Dict, NamedTuple, Tuple

import pandas as pd
from pydantic import BaseModel

import flyte
from flyte.io import DataFrame as FlyteDataFrame

env = flyte.TaskEnvironment(
    name="repro",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("pandas", "pyarrow"),
)


class MyModel(BaseModel):
    """Pydantic model wrapping a FlyteDataFrame."""

    data: FlyteDataFrame
    model_config = {"arbitrary_types_allowed": True}


@dataclass
class MyDataclass:
    """Dataclass wrapping a FlyteDataFrame."""

    data: FlyteDataFrame


class MyNamedTuple(NamedTuple):
    """NamedTuple wrapping a FlyteDataFrame."""

    data: FlyteDataFrame


@env.task
async def inner(model: MyModel) -> str:
    """Receives nested FlyteDataFrame in Pydantic model."""
    return "OK"


@env.task
async def inner_bare(data: FlyteDataFrame) -> str:
    """Receives bare FlyteDataFrame."""
    return "OK"


@env.task
async def inner_dataclass(model: MyDataclass) -> str:
    """Receives nested FlyteDataFrame in dataclass."""
    return "OK"


@env.task
async def inner_tuple(data: Tuple[FlyteDataFrame, str]) -> str:
    """Receives nested FlyteDataFrame in tuple."""
    return "OK"


@env.task
async def inner_dict(data: Dict[str, FlyteDataFrame]) -> str:
    """Receives nested FlyteDataFrame in dict."""
    return "OK"


@env.task
async def inner_namedtuple(model: MyNamedTuple) -> str:
    """Receives nested FlyteDataFrame in NamedTuple."""
    return "OK"


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)

    df = pd.DataFrame({"a": [1, 2, 3]})
    print("syncing df")
    fdf = FlyteDataFrame.from_local_sync(df)
    print("df synced")

    # Determine which test case to run based on command line args
    test_case = "pydantic"  # default
    if "--bare" in sys.argv:
        test_case = "bare"
    elif "--dataclass" in sys.argv:
        test_case = "dataclass"
    elif "--tuple" in sys.argv:
        test_case = "tuple"
    elif "--dict" in sys.argv:
        test_case = "dict"
    elif "--namedtuple" in sys.argv:
        test_case = "namedtuple"
    elif "--all" in sys.argv:
        test_case = "all"

    if test_case in ["bare", "all"]:
        print("Calling flyte.run() with BARE FlyteDataFrame...")
        run = flyte.run(inner_bare, data=fdf)
        print(f"SUCCESS (bare): {run}")

    if test_case in ["pydantic", "all"]:
        print("Calling flyte.run() with NESTED Pydantic model...")
        run = flyte.run(inner, model=MyModel(data=fdf))
        print(f"SUCCESS (pydantic): {run}")

    if test_case in ["dataclass", "all"]:
        print("Calling flyte.run() with NESTED dataclass...")
        run = flyte.run(inner_dataclass, model=MyDataclass(data=fdf))
        print(f"SUCCESS (dataclass): {run}")

    if test_case in ["tuple", "all"]:
        print("Calling flyte.run() with NESTED tuple...")
        run = flyte.run(inner_tuple, data=(fdf, "extra"))
        print(f"SUCCESS (tuple): {run}")

    if test_case in ["dict", "all"]:
        print("Calling flyte.run() with NESTED dict...")
        run = flyte.run(inner_dict, data={"my_df": fdf})
        print(f"SUCCESS (dict): {run}")

    if test_case in ["namedtuple", "all"]:
        print("Calling flyte.run() with NESTED NamedTuple...")
        run = flyte.run(inner_namedtuple, model=MyNamedTuple(data=fdf))
        print(f"SUCCESS (namedtuple): {run}")

    print("\n=== ALL TESTS PASSED ===")

    signal.alarm(0)
