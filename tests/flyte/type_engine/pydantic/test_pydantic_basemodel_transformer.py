from __future__ import annotations

import os
import tempfile
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import pytest
from flyteidl2.core.literals_pb2 import Literal, Scalar
from flyteidl2.core.types_pb2 import TypeAnnotation
from google.protobuf import json_format as _json_format
from google.protobuf import struct_pb2 as _struct
from pydantic import BaseModel, Field

import flyte
from flyte.io import Dir, File
from flyte.storage import S3
from flyte.types._type_engine import (
    CACHE_KEY_METADATA,
    MESSAGEPACK,
    SERIALIZATION_FORMAT,
    TypeEngine,
)


class Status(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@pytest.fixture
def local_dummy_file():
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp:
            tmp.write("Hello File")
        yield path
    finally:
        os.remove(path)


@pytest.fixture
def local_dummy_directory():
    temp_dir = tempfile.TemporaryDirectory()
    try:
        with open(os.path.join(temp_dir.name, "file"), "w") as tmp:
            tmp.write("Hello Dir")
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_flytetypes_in_pydantic_basemodel_wf(local_dummy_file, local_dummy_directory):
    temp_dir = tempfile.mkdtemp()
    flyte.init(storage=S3.for_sandbox(), root_dir=temp_dir)
    env = flyte.TaskEnvironment(name="pydantic-basemodel-transformer")

    class InnerBM(BaseModel):
        file: File = Field(default_factory=lambda: File(path=local_dummy_file))
        dir: Dir = Field(default_factory=lambda: Dir(path=local_dummy_directory))

    class BM(BaseModel):
        file: File = Field(default_factory=lambda: File(path=local_dummy_file))
        dir: Dir = Field(default_factory=lambda: Dir(path=local_dummy_directory))
        inner_bm: InnerBM = Field(default_factory=lambda: InnerBM())

    @env.task
    async def t1(path: File) -> File:
        return path

    @env.task
    async def t2(path: Dir) -> Dir:
        return path

    @env.task
    async def main(bm: BM) -> Tuple[File, File, Dir, Dir]:
        return (
            await t1(path=bm.file),
            await t1(path=bm.inner_bm.file),
            await t2(path=bm.dir),
            await t2(path=bm.inner_bm.dir),
        )

    o = flyte.run(main, bm=BM())
    o1, o2, o3, o4 = o.outputs()

    async with o1.open() as fh:
        content = await fh.read()
        content = content.decode("utf-8")
        assert content == "Hello File"

    async with o2.open() as fh:
        content = await fh.read()
        content = content.decode("utf-8")
        assert content == "Hello File"

    async for f in o3.walk():
        assert f.name == "file"
        async with f.open() as fh:
            content = await fh.read()
            content = content.decode("utf-8")
            assert content == "Hello Dir"

    async for f in o4.walk():
        assert f.name == "file"
        async with f.open() as fh:
            content = await fh.read()
            content = content.decode("utf-8")
            assert content == "Hello Dir"


@pytest.mark.asyncio
async def test_all_types_in_pydantic_basemodel_wf(local_dummy_file, local_dummy_directory):
    temp_dir = tempfile.mkdtemp()
    flyte.init(storage=S3.for_sandbox(), root_dir=temp_dir)
    env = flyte.TaskEnvironment(name="pydantic-basemodel-transformer")

    class InnerBM(BaseModel):
        a: int = -1
        b: float = 2.1
        c: str = "Hello, Flyte"
        d: bool = False
        e: List[int] = Field(default_factory=lambda: [0, 1, 2, -1, -2])
        f: List[File] = Field(default_factory=lambda: [File(path=local_dummy_file)])
        g: List[List[int]] = Field(default_factory=lambda: [[0], [1], [-1]])
        h: List[Dict[int, bool]] = Field(default_factory=lambda: [{0: False}, {1: True}, {-1: True}])
        i: Dict[int, bool] = Field(default_factory=lambda: {0: False, 1: True, -1: False})
        j: Dict[int, File] = Field(
            default_factory=lambda: {
                0: File(path=local_dummy_file),
                1: File(path=local_dummy_file),
                -1: File(path=local_dummy_file),
            }
        )
        k: Dict[int, List[int]] = Field(default_factory=lambda: {0: [0, 1, -1]})
        l: Dict[int, Dict[int, int]] = Field(default_factory=lambda: {1: {-1: 0}})  # noqa: E741
        m: dict = Field(default_factory=lambda: {"key": "value"})
        n: File = Field(default_factory=lambda: File(path=local_dummy_file))
        o: Dir = Field(default_factory=lambda: Dir(path=local_dummy_directory))
        enum_status: Status = Field(default=Status.PENDING)

    class BM(BaseModel):
        a: int = -1
        b: float = 2.1
        c: str = "Hello, Flyte"
        d: bool = False
        e: List[int] = Field(default_factory=lambda: [0, 1, 2, -1, -2])
        f: List[File] = Field(
            default_factory=lambda: [
                File(path=local_dummy_file),
            ]
        )
        g: List[List[int]] = Field(default_factory=lambda: [[0], [1], [-1]])
        h: List[Dict[int, bool]] = Field(default_factory=lambda: [{0: False}, {1: True}, {-1: True}])
        i: Dict[int, bool] = Field(default_factory=lambda: {0: False, 1: True, -1: False})
        j: Dict[int, File] = Field(
            default_factory=lambda: {
                0: File(path=local_dummy_file),
                1: File(path=local_dummy_file),
                -1: File(path=local_dummy_file),
            }
        )
        k: Dict[int, List[int]] = Field(default_factory=lambda: {0: [0, 1, -1]})
        l: Dict[int, Dict[int, int]] = Field(default_factory=lambda: {1: {-1: 0}})  # noqa: E741
        m: dict = Field(default_factory=lambda: {"key": "value"})
        n: File = Field(default_factory=lambda: File(path=local_dummy_file))
        o: Dir = Field(default_factory=lambda: Dir(path=local_dummy_directory))
        inner_bm: InnerBM = Field(default_factory=lambda: InnerBM())
        enum_status: Status = Field(default=Status.PENDING)

    @env.task
    async def t_inner(inner_bm: InnerBM):
        assert type(inner_bm) is InnerBM

        # f: List[File]
        for ff in inner_bm.f:
            assert type(ff) is File
            async with ff.open() as f:
                assert (await f.read()).decode("utf-8") == "Hello File"
        # j: Dict[int, File]
        for _, ff in inner_bm.j.items():
            assert type(ff) is File
            async with ff.open() as f:
                assert (await f.read()).decode("utf-8") == "Hello File"
        # n: File
        assert type(inner_bm.n) is File
        async with inner_bm.n.open() as f:
            assert (await f.read()).decode("utf-8") == "Hello File"
        # o: Dir
        assert type(inner_bm.o) is Dir
        async for f in inner_bm.o.walk():
            assert f.name == "file"
            async with f.open() as fh:
                content = await fh.read()
                content = content.decode("utf-8")
                assert content == "Hello Dir"

        # enum: Status
        assert inner_bm.enum_status == Status.PENDING

    @env.task
    async def t_test_all_attributes(
        a: int,
        b: float,
        c: str,
        d: bool,
        e: List[int],
        f: List[File],
        g: List[List[int]],
        h: List[Dict[int, bool]],
        i: Dict[int, bool],
        j: Dict[int, File],
        k: Dict[int, List[int]],
        l: Dict[int, Dict[int, int]],  # noqa: E741
        m: dict,
        n: File,
        o: Dir,
        enum_status: Status,
    ):
        # Strict type checks for simple types
        assert isinstance(a, int), f"a is not int, it's {type(a)}"
        assert a == -1
        assert isinstance(b, float), f"b is not float, it's {type(b)}"
        assert isinstance(c, str), f"c is not str, it's {type(c)}"
        assert isinstance(d, bool), f"d is not bool, it's {type(d)}"

        # Strict type checks for List[int]
        assert isinstance(e, list) and all(isinstance(i, int) for i in e), "e is not List[int]"

        # Strict type checks for List[File]
        assert isinstance(f, list) and all(isinstance(i, File) for i in f), "f is not List[File]"

        # Strict type checks for List[List[int]]
        assert isinstance(g, list) and all(isinstance(i, list) and all(isinstance(j, int) for j in i) for i in g), (
            "g is not List[List[int]]"
        )

        # Strict type checks for List[Dict[int, bool]]
        assert isinstance(h, list) and all(
            isinstance(i, dict) and all(isinstance(k, int) and isinstance(v, bool) for k, v in i.items()) for i in h
        ), "h is not List[Dict[int, bool]]"

        # Strict type checks for Dict[int, bool]
        assert isinstance(i, dict) and all(isinstance(k, int) and isinstance(v, bool) for k, v in i.items()), (
            "i is not Dict[int, bool]"
        )

        # Strict type checks for Dict[int, File]
        assert isinstance(j, dict) and all(isinstance(k, int) and isinstance(v, File) for k, v in j.items()), (
            "j is not Dict[int, File]"
        )

        # Strict type checks for Dict[int, List[int]]
        assert isinstance(k, dict) and all(
            isinstance(k, int) and isinstance(v, list) and all(isinstance(i, int) for i in v) for k, v in k.items()
        ), "k is not Dict[int, List[int]]"

        # Strict type checks for Dict[int, Dict[int, int]]
        assert isinstance(l, dict) and all(
            isinstance(k, int)
            and isinstance(v, dict)
            and all(isinstance(sub_k, int) and isinstance(sub_v, int) for sub_k, sub_v in v.items())
            for k, v in l.items()
        ), "l is not Dict[int, Dict[int, int]]"

        # Strict type check for a generic dict
        assert isinstance(m, dict), "m is not dict"

        # Strict type check for File
        assert isinstance(n, File), "n is not File"

        # Strict type check for Dir
        assert isinstance(o, Dir), "o is not Dir"

        # Strict type check for Enum
        assert isinstance(enum_status, Status), "enum_status is not Status"

        print("All attributes passed strict type checks.")

    @env.task
    async def main(bm: BM):
        await t_inner(bm.inner_bm)
        await t_test_all_attributes(
            a=bm.a,
            b=bm.b,
            c=bm.c,
            d=bm.d,
            e=bm.e,
            f=bm.f,
            g=bm.g,
            h=bm.h,
            i=bm.i,
            j=bm.j,
            k=bm.k,
            l=bm.l,
            m=bm.m,
            n=bm.n,
            o=bm.o,
            enum_status=bm.enum_status,
        )
        await t_test_all_attributes(
            a=bm.inner_bm.a,
            b=bm.inner_bm.b,
            c=bm.inner_bm.c,
            d=bm.inner_bm.d,
            e=bm.inner_bm.e,
            f=bm.inner_bm.f,
            g=bm.inner_bm.g,
            h=bm.inner_bm.h,
            i=bm.inner_bm.i,
            j=bm.inner_bm.j,
            k=bm.inner_bm.k,
            l=bm.inner_bm.l,
            m=bm.inner_bm.m,
            n=bm.inner_bm.n,
            o=bm.inner_bm.o,
            enum_status=bm.inner_bm.enum_status,
        )

    flyte.run(main, bm=BM())


@pytest.mark.asyncio
async def test_all_types_with_optional_in_pydantic_basemodel_wf(local_dummy_file, local_dummy_directory):
    temp_dir = tempfile.mkdtemp()
    flyte.init(storage=S3.for_sandbox(), root_dir=temp_dir)
    env = flyte.TaskEnvironment(name="pydantic-basemodel-transformer")

    class InnerBM(BaseModel):
        a: Optional[int] = -1
        b: Optional[float] = 2.1
        c: Optional[str] = "Hello, Flyte"
        d: Optional[bool] = False
        e: Optional[List[int]] = Field(default_factory=lambda: [0, 1, 2, -1, -2])
        f: Optional[List[File]] = Field(default_factory=lambda: [File(path=local_dummy_file)])
        g: Optional[List[List[int]]] = Field(default_factory=lambda: [[0], [1], [-1]])
        h: Optional[List[Dict[int, bool]]] = Field(default_factory=lambda: [{0: False}, {1: True}, {-1: True}])
        i: Optional[Dict[int, bool]] = Field(default_factory=lambda: {0: False, 1: True, -1: False})
        j: Optional[Dict[int, File]] = Field(
            default_factory=lambda: {
                0: File(path=local_dummy_file),
                1: File(path=local_dummy_file),
                -1: File(path=local_dummy_file),
            }
        )
        k: Optional[Dict[int, List[int]]] = Field(default_factory=lambda: {0: [0, 1, -1]})
        l: Optional[Dict[int, Dict[int, int]]] = Field(default_factory=lambda: {1: {-1: 0}})  # noqa: E741
        m: Optional[dict] = Field(default_factory=lambda: {"key": "value"})
        n: Optional[File] = Field(default_factory=lambda: File(path=local_dummy_file))
        o: Optional[Dir] = Field(default_factory=lambda: Dir(path=local_dummy_directory))
        enum_status: Optional[Status] = Field(default=Status.PENDING)

    class BM(BaseModel):
        a: Optional[int] = -1
        b: Optional[float] = 2.1
        c: Optional[str] = "Hello, Flyte"
        d: Optional[bool] = False
        e: Optional[List[int]] = Field(default_factory=lambda: [0, 1, 2, -1, -2])
        f: Optional[List[File]] = Field(default_factory=lambda: [File(path=local_dummy_file)])
        g: Optional[List[List[int]]] = Field(default_factory=lambda: [[0], [1], [-1]])
        h: Optional[List[Dict[int, bool]]] = Field(default_factory=lambda: [{0: False}, {1: True}, {-1: True}])
        i: Optional[Dict[int, bool]] = Field(default_factory=lambda: {0: False, 1: True, -1: False})
        j: Optional[Dict[int, File]] = Field(
            default_factory=lambda: {
                0: File(path=local_dummy_file),
                1: File(path=local_dummy_file),
                -1: File(path=local_dummy_file),
            }
        )
        k: Optional[Dict[int, List[int]]] = Field(default_factory=lambda: {0: [0, 1, -1]})
        l: Optional[Dict[int, Dict[int, int]]] = Field(default_factory=lambda: {1: {-1: 0}})  # noqa: E741
        m: Optional[dict] = Field(default_factory=lambda: {"key": "value"})
        n: Optional[File] = Field(default_factory=lambda: File(path=local_dummy_file))
        o: Optional[Dir] = Field(default_factory=lambda: Dir(path=local_dummy_directory))
        inner_bm: Optional[InnerBM] = Field(default_factory=lambda: InnerBM())
        enum_status: Optional[Status] = Field(default=Status.PENDING)

    @env.task
    async def t_inner(inner_bm: InnerBM):
        assert type(inner_bm) is InnerBM

        # f: List[File]
        for ff in inner_bm.f:
            assert type(ff) is File
            async with ff.open() as f:
                assert (await f.read()).decode("utf-8") == "Hello File"
        # j: Dict[int, File]
        for _, ff in inner_bm.j.items():
            assert type(ff) is File
            async with ff.open() as f:
                assert (await f.read()).decode("utf-8") == "Hello File"
        # n: File
        assert type(inner_bm.n) is File
        async with inner_bm.n.open() as f:
            assert (await f.read()).decode("utf-8") == "Hello File"
        # o: Dir
        assert type(inner_bm.o) is Dir
        async for f in inner_bm.o.walk():
            assert f.name == "file"
            async with f.open() as fh:
                content = await fh.read()
                content = content.decode("utf-8")
                assert content == "Hello Dir"

        # enum: Status
        assert inner_bm.enum_status == Status.PENDING

    @env.task
    async def t_test_all_attributes(
        a: Optional[int],
        b: Optional[float],
        c: Optional[str],
        d: Optional[bool],
        e: Optional[List[int]],
        f: Optional[List[File]],
        g: Optional[List[List[int]]],
        h: Optional[List[Dict[int, bool]]],
        i: Optional[Dict[int, bool]],
        j: Optional[Dict[int, File]],
        k: Optional[Dict[int, List[int]]],
        l: Optional[Dict[int, Dict[int, int]]],  # noqa: E741
        m: Optional[dict],
        n: Optional[File],
        o: Optional[Dir],
        enum_status: Optional[Status],
    ):
        # Strict type checks for simple types
        assert isinstance(a, int), f"a is not int, it's {type(a)}"
        assert a == -1
        assert isinstance(b, float), f"b is not float, it's {type(b)}"
        assert isinstance(c, str), f"c is not str, it's {type(c)}"
        assert isinstance(d, bool), f"d is not bool, it's {type(d)}"

        # Strict type checks for List[int]
        assert isinstance(e, list) and all(isinstance(i, int) for i in e), "e is not List[int]"

        # Strict type checks for List[File]
        assert isinstance(f, list) and all(isinstance(i, File) for i in f), "f is not List[File]"

        # Strict type checks for List[List[int]]
        assert isinstance(g, list) and all(isinstance(i, list) and all(isinstance(j, int) for j in i) for i in g), (
            "g is not List[List[int]]"
        )

        # Strict type checks for List[Dict[int, bool]]
        assert isinstance(h, list) and all(
            isinstance(i, dict) and all(isinstance(k, int) and isinstance(v, bool) for k, v in i.items()) for i in h
        ), "h is not List[Dict[int, bool]]"

        # Strict type checks for Dict[int, bool]
        assert isinstance(i, dict) and all(isinstance(k, int) and isinstance(v, bool) for k, v in i.items()), (
            "i is not Dict[int, bool]"
        )

        # Strict type checks for Dict[int, File]
        assert isinstance(j, dict) and all(isinstance(k, int) and isinstance(v, File) for k, v in j.items()), (
            "j is not Dict[int, File]"
        )

        # Strict type checks for Dict[int, List[int]]
        assert isinstance(k, dict) and all(
            isinstance(k, int) and isinstance(v, list) and all(isinstance(i, int) for i in v) for k, v in k.items()
        ), "k is not Dict[int, List[int]]"

        # Strict type checks for Dict[int, Dict[int, int]]
        assert isinstance(l, dict) and all(
            isinstance(k, int)
            and isinstance(v, dict)
            and all(isinstance(sub_k, int) and isinstance(sub_v, int) for sub_k, sub_v in v.items())
            for k, v in l.items()
        ), "l is not Dict[int, Dict[int, int]]"

        # Strict type check for a generic dict
        assert isinstance(m, dict), "m is not dict"

        # Strict type check for File
        assert isinstance(n, File), "n is not File"

        # Strict type check for Dir
        assert isinstance(o, Dir), "o is not Dir"

        # Strict type check for Enum
        assert isinstance(enum_status, Status), "enum_status is not Status"

    @env.task
    async def main(bm: BM):
        await t_inner(bm.inner_bm)
        await t_test_all_attributes(
            a=bm.a,
            b=bm.b,
            c=bm.c,
            d=bm.d,
            e=bm.e,
            f=bm.f,
            g=bm.g,
            h=bm.h,
            i=bm.i,
            j=bm.j,
            k=bm.k,
            l=bm.l,
            m=bm.m,
            n=bm.n,
            o=bm.o,
            enum_status=bm.enum_status,
        )

    flyte.run(main, bm=BM())


@pytest.mark.asyncio
async def test_all_types_with_optional_and_none_in_pydantic_basemodel_wf():
    env = flyte.TaskEnvironment(name="pydantic-basemodel-transformer")

    class InnerBM(BaseModel):
        a: Optional[int] = None
        b: Optional[float] = None
        c: Optional[str] = None
        d: Optional[bool] = None
        e: Optional[List[int]] = None
        f: Optional[List[File]] = None
        g: Optional[List[List[int]]] = None
        h: Optional[List[Dict[int, bool]]] = None
        i: Optional[Dict[int, bool]] = None
        j: Optional[Dict[int, File]] = None
        k: Optional[Dict[int, List[int]]] = None
        l: Optional[Dict[int, Dict[int, int]]] = None  # noqa: E741
        m: Optional[dict] = None
        n: Optional[File] = None
        o: Optional[Dir] = None
        enum_status: Optional[Status] = None

    class BM(BaseModel):
        a: Optional[int] = None
        b: Optional[float] = None
        c: Optional[str] = None
        d: Optional[bool] = None
        e: Optional[List[int]] = None
        f: Optional[List[File]] = None
        g: Optional[List[List[int]]] = None
        h: Optional[List[Dict[int, bool]]] = None
        i: Optional[Dict[int, bool]] = None
        j: Optional[Dict[int, File]] = None
        k: Optional[Dict[int, List[int]]] = None
        l: Optional[Dict[int, Dict[int, int]]] = None  # noqa: E741
        m: Optional[dict] = None
        n: Optional[File] = None
        o: Optional[Dir] = None
        inner_bm: Optional[InnerBM] = None
        enum_status: Optional[Status] = None

    @env.task
    async def t_inner(inner_bm: Optional[InnerBM]) -> Optional[InnerBM]:
        return inner_bm

    @env.task
    async def t_test_all_attributes(
        a: Optional[int],
        b: Optional[float],
        c: Optional[str],
        d: Optional[bool],
        e: Optional[List[int]],
        f: Optional[List[File]],
        g: Optional[List[List[int]]],
        h: Optional[List[Dict[int, bool]]],
        i: Optional[Dict[int, bool]],
        j: Optional[Dict[int, File]],
        k: Optional[Dict[int, List[int]]],
        l: Optional[Dict[int, Dict[int, int]]],  # noqa: E741
        m: Optional[dict],
        n: Optional[File],
        o: Optional[Dir],
        enum_status: Optional[Status],
    ):
        return

    bm = BM()

    await t_inner(bm.inner_bm)
    await t_test_all_attributes(
        a=bm.a,
        b=bm.b,
        c=bm.c,
        d=bm.d,
        e=bm.e,
        f=bm.f,
        g=bm.g,
        h=bm.h,
        i=bm.i,
        j=bm.j,
        k=bm.k,
        l=bm.l,
        m=bm.m,
        n=bm.n,
        o=bm.o,
        enum_status=bm.enum_status,
    )


@pytest.mark.asyncio
async def test_input_from_ui_pydantic_basemodel(local_dummy_file, local_dummy_directory):
    # UI will send the input data as protobuf Struct

    temp_dir = tempfile.mkdtemp()
    flyte.init(storage=S3.for_sandbox(), root_dir=temp_dir)
    env = flyte.TaskEnvironment(name="pydantic-basemodel-transformer")

    class InnerBM(BaseModel):
        a: int = -1
        b: float = 2.1
        c: str = "Hello, Flyte"
        d: bool = False
        e: List[int] = Field(default_factory=lambda: [0, 1, 2, -1, -2])
        f: List[File] = Field(default_factory=lambda: [File(path=local_dummy_file)])
        g: List[List[int]] = Field(default_factory=lambda: [[0], [1], [-1]])
        h: List[Dict[int, bool]] = Field(default_factory=lambda: [{0: False}, {1: True}, {-1: True}])
        i: Dict[int, bool] = Field(default_factory=lambda: {0: False, 1: True, -1: False})
        j: Dict[int, File] = Field(
            default_factory=lambda: {
                0: File(path=local_dummy_file),
                1: File(path=local_dummy_file),
                -1: File(path=local_dummy_file),
            }
        )
        k: Dict[int, List[int]] = Field(default_factory=lambda: {0: [0, 1, -1]})
        l: Dict[int, Dict[int, int]] = Field(default_factory=lambda: {1: {-1: 0}})  # noqa: E741
        m: dict = Field(default_factory=lambda: {"key": "value"})
        n: File = Field(default_factory=lambda: File(path=local_dummy_file))
        o: Dir = Field(default_factory=lambda: Dir(path=local_dummy_directory))
        enum_status: Status = Field(default=Status.PENDING)

    class BM(BaseModel):
        a: int = -1
        b: float = 2.1
        c: str = "Hello, Flyte"
        d: bool = False
        e: List[int] = Field(default_factory=lambda: [0, 1, 2, -1, -2])
        f: List[File] = Field(
            default_factory=lambda: [
                File(path=local_dummy_file),
            ]
        )
        g: List[List[int]] = Field(default_factory=lambda: [[0], [1], [-1]])
        h: List[Dict[int, bool]] = Field(default_factory=lambda: [{0: False}, {1: True}, {-1: True}])
        i: Dict[int, bool] = Field(default_factory=lambda: {0: False, 1: True, -1: False})
        j: Dict[int, File] = Field(
            default_factory=lambda: {
                0: File(path=local_dummy_file),
                1: File(path=local_dummy_file),
                -1: File(path=local_dummy_file),
            }
        )
        k: Dict[int, List[int]] = Field(default_factory=lambda: {0: [0, 1, -1]})
        l: Dict[int, Dict[int, int]] = Field(default_factory=lambda: {1: {-1: 0}})  # noqa: E741
        m: dict = Field(default_factory=lambda: {"key": "value"})
        n: File = Field(default_factory=lambda: File(path=local_dummy_file))
        o: Dir = Field(default_factory=lambda: Dir(path=local_dummy_directory))
        inner_bm: InnerBM = Field(default_factory=lambda: InnerBM())
        enum_status: Status = Field(default=Status.PENDING)

    @env.task
    async def t_inner(inner_bm: InnerBM):
        assert type(inner_bm) is InnerBM

        # f: List[File]
        for ff in inner_bm.f:
            assert type(ff) is File
            async with ff.open() as f:
                assert (await f.read()).decode("utf-8") == "Hello File"
        # j: Dict[int, File]
        for _, ff in inner_bm.j.items():
            assert type(ff) is File
            async with ff.open() as f:
                assert (await f.read()).decode("utf-8") == "Hello File"
        # n: File
        assert type(inner_bm.n) is File
        async with inner_bm.n.open() as f:
            assert (await f.read()).decode("utf-8") == "Hello File"
        # o: Dir
        assert type(inner_bm.o) is Dir
        async for f in inner_bm.o.walk():
            assert f.name == "file"
            async with f.open() as fh:
                content = await fh.read()
                content = content.decode("utf-8")
                assert content == "Hello Dir"

        # enum: Status
        assert inner_bm.enum_status == Status.PENDING

    @env.task
    async def t_test_all_attributes(
        a: int,
        b: float,
        c: str,
        d: bool,
        e: List[int],
        f: List[File],
        g: List[List[int]],
        h: List[Dict[int, bool]],
        i: Dict[int, bool],
        j: Dict[int, File],
        k: Dict[int, List[int]],
        l: Dict[int, Dict[int, int]],  # noqa: E741
        m: dict,
        n: File,
        o: Dir,
        enum_status: Status,
    ):
        # Strict type checks for simple types
        assert isinstance(a, int), f"a is not int, it's {type(a)}"
        assert a == -1
        assert isinstance(b, float), f"b is not float, it's {type(b)}"
        assert isinstance(c, str), f"c is not str, it's {type(c)}"
        assert isinstance(d, bool), f"d is not bool, it's {type(d)}"

        # Strict type checks for List[int]
        assert isinstance(e, list) and all(isinstance(i, int) for i in e), "e is not List[int]"

        # Strict type checks for List[File]
        assert isinstance(f, list) and all(isinstance(i, File) for i in f), "f is not List[File]"

        # Strict type checks for List[List[int]]
        assert isinstance(g, list) and all(isinstance(i, list) and all(isinstance(j, int) for j in i) for i in g), (
            "g is not List[List[int]]"
        )

        # Strict type checks for List[Dict[int, bool]]
        assert isinstance(h, list) and all(
            isinstance(i, dict) and all(isinstance(k, int) and isinstance(v, bool) for k, v in i.items()) for i in h
        ), "h is not List[Dict[int, bool]]"

        # Strict type checks for Dict[int, bool]
        assert isinstance(i, dict) and all(isinstance(k, int) and isinstance(v, bool) for k, v in i.items()), (
            "i is not Dict[int, bool]"
        )

        # Strict type checks for Dict[int, File]
        assert isinstance(j, dict) and all(isinstance(k, int) and isinstance(v, File) for k, v in j.items()), (
            "j is not Dict[int, File]"
        )

        # Strict type checks for Dict[int, List[int]]
        assert isinstance(k, dict) and all(
            isinstance(k, int) and isinstance(v, list) and all(isinstance(i, int) for i in v) for k, v in k.items()
        ), "k is not Dict[int, List[int]]"

        # Strict type checks for Dict[int, Dict[int, int]]
        assert isinstance(l, dict) and all(
            isinstance(k, int)
            and isinstance(v, dict)
            and all(isinstance(sub_k, int) and isinstance(sub_v, int) for sub_k, sub_v in v.items())
            for k, v in l.items()
        ), "l is not Dict[int, Dict[int, int]]"

        # Strict type check for a generic dict
        assert isinstance(m, dict), "m is not dict"

        # Strict type check for File
        assert isinstance(n, File), "n is not File"

        # Strict type check for Dir
        assert isinstance(o, Dir), "o is not Dir"

        # Strict type check for Enum
        assert isinstance(enum_status, Status), "enum_status is not Status"

        print("All attributes passed strict type checks.")

    bm = BM()
    json_str = bm.model_dump_json()
    upstream_output = Literal(scalar=Scalar(generic=_json_format.Parse(json_str, _struct.Struct())))

    downstream_input = await TypeEngine.to_python_value(upstream_output, BM)
    flyte.run(t_inner, inner_bm=downstream_input.inner_bm)
    flyte.run(
        t_test_all_attributes,
        a=downstream_input.a,
        b=downstream_input.b,
        c=downstream_input.c,
        d=downstream_input.d,
        e=downstream_input.e,
        f=downstream_input.f,
        g=downstream_input.g,
        h=downstream_input.h,
        i=downstream_input.i,
        j=downstream_input.j,
        k=downstream_input.k,
        l=downstream_input.l,
        m=downstream_input.m,
        n=downstream_input.n,
        o=downstream_input.o,
        enum_status=downstream_input.enum_status,
    )
    flyte.run(
        t_test_all_attributes,
        a=downstream_input.inner_bm.a,
        b=downstream_input.inner_bm.b,
        c=downstream_input.inner_bm.c,
        d=downstream_input.inner_bm.d,
        e=downstream_input.inner_bm.e,
        f=downstream_input.inner_bm.f,
        g=downstream_input.inner_bm.g,
        h=downstream_input.inner_bm.h,
        i=downstream_input.inner_bm.i,
        j=downstream_input.inner_bm.j,
        k=downstream_input.inner_bm.k,
        l=downstream_input.inner_bm.l,
        m=downstream_input.inner_bm.m,
        n=downstream_input.inner_bm.n,
        o=downstream_input.inner_bm.o,
        enum_status=downstream_input.inner_bm.enum_status,
    )


@pytest.mark.asyncio
async def test_union_in_basemodel_wf():
    temp_dir = tempfile.mkdtemp()
    flyte.init(storage=S3.for_sandbox(), root_dir=temp_dir)
    env = flyte.TaskEnvironment(name="pydantic-basemodel-transformer")

    class BM(BaseModel):
        a: Union[int, bool, str, float]
        b: Union[int, bool, str, float]

    @env.task
    async def add(a: Union[int, bool, str, float], b: Union[int, bool, str, float]) -> Union[int, bool, str, float]:
        return a + b  # type: ignore

    @env.task
    async def main(bm: BM) -> Union[int, bool, str, float]:
        return await add(bm.a, bm.b)

    assert flyte.run(main, bm=BM(a=1, b=2)).outputs() == 3
    assert flyte.run(main, bm=BM(a=True, b=False)).outputs() == 1
    assert flyte.run(main, bm=BM(a=False, b=False)).outputs() == 0
    assert flyte.run(main, bm=BM(a="hello", b="world")).outputs() == "helloworld"
    assert flyte.run(main, bm=BM(a=1.0, b=2.0)).outputs() == 3.0

    @env.task
    async def add_bm(bm1: BM, bm2: BM) -> Union[int, bool, str, float]:
        return bm1.a + bm2.b  # type: ignore

    bm = BM(a=1, b=2)
    assert flyte.run(add_bm, bm1=bm, bm2=bm).outputs() == 3

    @env.task
    async def return_bm(bm: BM) -> BM:
        return bm

    assert flyte.run(return_bm, bm=BM(a=1, b=2)).outputs() == BM(a=1, b=2)


@pytest.mark.asyncio
async def test_basemodel_literal_type_annotation():
    class BM(BaseModel):
        a: int = -1
        b: float = 2.1
        c: str = "Hello, Flyte"

    assert TypeEngine.to_literal_type(BM).annotation == TypeAnnotation(
        annotations={CACHE_KEY_METADATA: {SERIALIZATION_FORMAT: MESSAGEPACK}}
    )


@pytest.mark.asyncio
async def test_basic_pydantic_type_engine():
    class DC(BaseModel):
        string: Optional[str] = None

    class PydWithOptional(BaseModel):
        string: Optional[str] = None
        dc: Optional[DC] = None
        list_dc: Optional[List[DC]] = None
        list_list_dc: Optional[List[List[DC]]] = None
        dict_dc: Optional[Dict[str, DC]] = None
        dict_dict_dc: Optional[Dict[str, Dict[str, DC]]] = None
        dict_list_dc: Optional[Dict[str, List[DC]]] = None
        list_dict_dc: Optional[List[Dict[str, DC]]] = None

    lt = TypeEngine.to_literal_type(PydWithOptional)
    assert lt.simple == 9

    p = PydWithOptional(
        string="test",
        dc=DC(string="dc_test"),
        list_dc=[DC(string="list_dc_test")],
        list_list_dc=[[DC(string="list_list_dc_test")]],
        dict_dc={"key1": DC(string="dict_dc_test")},
        dict_dict_dc={"key1": {"sub_key1": DC(string="dict_dict_dc_test")}},
        dict_list_dc={"key1": [DC(string="dict_list_dc_test")]},
        list_dict_dc=[{"key1": DC(string="list_dict_dc_test")}],
    )
    lit = await TypeEngine.to_literal(p, PydWithOptional, lt)
    assert lit.scalar.HasField("binary")

    pv = await TypeEngine.to_python_value(lit, PydWithOptional)
    assert pv == p
