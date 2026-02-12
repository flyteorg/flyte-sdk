import asyncio
from typing import Dict, List, Optional

from flyteidl2.core.literals_pb2 import Literal
from flyteidl2.core.types_pb2 import LiteralType, SimpleType
from pydantic import BaseModel

import flyte
from flyte.io import Dir, File
from flyte.types import TypeEngine, TypeTransformer

env = flyte.TaskEnvironment(name="inputs_pydantic_custom_type_in_nested")


class Coordinate(BaseModel):
    x: float
    y: float


class CoordinateTransformer(TypeTransformer[Coordinate]):
    """A custom transformer for Coordinate.

    Since Coordinate is a BaseModel subclass, schema_match works automatically
    — no override needed. This lets the type engine recognize Coordinate when
    it appears nested inside other Pydantic models.
    """

    def __init__(self):
        super().__init__("Coordinate", Coordinate)

    def get_literal_type(self, t=None) -> LiteralType:
        return LiteralType(simple=SimpleType.STRUCT)

    async def to_literal(self, python_val, python_type, expected) -> Literal:
        raise NotImplementedError

    async def to_python_value(self, lv, expected_python_type):
        raise NotImplementedError


TypeEngine.register(CoordinateTransformer())


class ModelWithCoord(BaseModel):
    label: str
    coord: Coordinate


class ModelWithListOfCoords(BaseModel):
    coords: List[Coordinate]


class ModelWithDictOfCoords(BaseModel):
    coord_map: Dict[str, Coordinate]


class ModelWithNestedListOfCoords(BaseModel):
    nested: List[List[Coordinate]]


class ModelWithOptionalCoord(BaseModel):
    coord: Optional[Coordinate] = None


class ModelWithMixed(BaseModel):
    coords: List[Coordinate]
    file: File
    dir_map: Dict[str, Dir]


@env.task
async def direct_coord(data: ModelWithCoord) -> str:
    return f"Label: {data.label}, Coord: ({data.coord.x}, {data.coord.y})"


@env.task
async def list_of_coords(data: ModelWithListOfCoords) -> str:
    points = [(c.x, c.y) for c in data.coords]
    return f"Coords ({len(points)}): {points}"


@env.task
async def dict_of_coords(data: ModelWithDictOfCoords) -> str:
    items = {k: (v.x, v.y) for k, v in data.coord_map.items()}
    return f"Coord map: {items}"


@env.task
async def nested_list_of_coords(data: ModelWithNestedListOfCoords) -> str:
    rows = [[(c.x, c.y) for c in row] for row in data.nested]
    return f"Nested coords ({len(rows)} rows): {rows}"


@env.task
async def optional_coord(data: ModelWithOptionalCoord) -> str:
    if data.coord is not None:
        return f"Coord: ({data.coord.x}, {data.coord.y})"
    return "Coord: None"


@env.task
async def mixed_types(data: ModelWithMixed) -> str:
    points = [(c.x, c.y) for c in data.coords]
    dirs = list(data.dir_map.keys())
    return f"Coords: {points}, File: {data.file.path}, Dirs: {dirs}"


@env.task
async def main() -> str:
    r1, r2, r3, r4, r5, r6, r7 = await asyncio.gather(
        direct_coord(
            data=ModelWithCoord(label="origin", coord=Coordinate(x=0.0, y=0.0)),
        ),
        list_of_coords(
            data=ModelWithListOfCoords(
                coords=[Coordinate(x=1.0, y=2.0), Coordinate(x=3.0, y=4.0)],
            ),
        ),
        dict_of_coords(
            data=ModelWithDictOfCoords(
                coord_map={"a": Coordinate(x=1.0, y=1.0), "b": Coordinate(x=2.0, y=2.0)},
            ),
        ),
        nested_list_of_coords(
            data=ModelWithNestedListOfCoords(
                nested=[[Coordinate(x=0.0, y=0.0)], [Coordinate(x=1.0, y=1.0), Coordinate(x=2.0, y=2.0)]],
            ),
        ),
        optional_coord(
            data=ModelWithOptionalCoord(coord=Coordinate(x=5.0, y=5.0)),
        ),
        optional_coord(
            data=ModelWithOptionalCoord(),
        ),
        mixed_types(
            data=ModelWithMixed(
                coords=[Coordinate(x=1.0, y=2.0)],
                file=File(path="example.txt"),
                dir_map={"d1": Dir(path="/tmp/d1")},
            ),
        ),
    )

    return f"{r1}\n{r2}\n{r3}\n{r4}\n{r5}\n{r6}\n{r7}"


if __name__ == "__main__":
    flyte.init_from_config()

    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()