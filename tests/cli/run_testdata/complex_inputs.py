import datetime
import enum
import typing
from dataclasses import dataclass

import flyte
from flyte.io import Dir, File

env = flyte.TaskEnvironment(name="hello_world")


@dataclass
class MyDataclass:
    i: int
    a: typing.List[str]


@dataclass
class NestedDataclass:
    i: typing.List[MyDataclass]


class Color(enum.Enum):
    RED = "RED"
    GREEN = "GREEN"
    BLUE = "BLUE"


@env.task
async def print_all(
    a: int,
    b: str,
    c: float,
    d: MyDataclass,
    e: typing.List[int],
    f: typing.Dict[str, float],
    g: File,
    h: bool,
    i: datetime.datetime,
    j: datetime.timedelta,
    k: Color,
    m: dict,
    # todo: add support for nested lists and dicts
    # # n: typing.List[typing.Dict[str, File]],
    # # o: typing.Dict[str, typing.List[File]],
    p: typing.Any,
    q: Dir,
    r: typing.List[MyDataclass],
    s: typing.Dict[str, MyDataclass],
    t: NestedDataclass,
):
    print(f"{a}, {b}, {c}, {d}, {e}, {f}, {g}, {h}, {i}, {j}, {k}, {m}, {p}, {q}, {r}, {s}, {t}")
