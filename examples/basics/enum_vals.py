import asyncio
import enum
import typing
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

import flyte


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


Intensity = Literal["low", "medium", "high"]


@dataclass
class DataPoint:
    pos: int
    color: Color
    intensity: Intensity


class PyDataPoint(BaseModel):
    pos: int
    color: Color
    intensity: Intensity


env = flyte.TaskEnvironment(name="enum_vals")


@env.task
async def enum_echo(c: Color) -> Color:
    return c


@env.task
async def enum_task(c: Color) -> str:
    return f"Color is {c.value}"


@env.task
async def literal_echo(i: Intensity) -> Intensity:
    return i


@env.task
async def literal_task(i: Intensity) -> str:
    return f"Intensity is {i}"


@env.task
async def dataclass_echo(dp: DataPoint) -> DataPoint:
    return dp


@env.task
async def dataclass_task(dp: DataPoint) -> str:
    return f"DataPoint is pos={dp.pos}, color={dp.color.value}, intensity={dp.intensity}"


@env.task
async def pydantic_echo(dp: PyDataPoint) -> PyDataPoint:
    return dp


@env.task
async def pydantic_task(dp: PyDataPoint) -> str:
    return f"PyDataPoint is pos={dp.pos}, color={dp.color.value}, intensity={dp.intensity}"


async def echo_pipe(x: typing.Any, src: typing.Callable, sink: typing.Callable) -> str:
    return await sink(await src(x))


@env.task
async def main() -> list[str]:
    res: list[typing.Coroutine[..., ..., str]] = []
    res.append(echo_pipe(Color.RED, enum_echo, enum_task))
    res.append(echo_pipe("high", literal_echo, literal_task))
    res.append(echo_pipe(DataPoint(pos=1, color=Color.GREEN, intensity="medium"), dataclass_echo, dataclass_task))
    res.append(echo_pipe(PyDataPoint(pos=2, color=Color.BLUE, intensity="low"), pydantic_echo, pydantic_task))
    return await asyncio.gather(*res)


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    r = flyte.run(main)
    print(r.url)
