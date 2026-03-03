from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pytest

import flyte
from flyte.io import Dir, File
from flyte.storage import S3
from flyte.types._type_engine import (
    TypeEngine,
)


@dataclass
class PrimitiveDC:
    int_field: int = 42
    float_field: float = 3.14
    str_field: str = "hello"
    bool_field: bool | None = None
    list_field: List[int] = None
    generic_list_field: list[str] = None


@pytest.mark.asyncio
async def test_basic_dc_typing(local_dummy_file, local_dummy_directory):
    literal_type = TypeEngine.to_literal_type(PrimitiveDC)

    primitive = PrimitiveDC(int_field=42, float_field=3.14, list_field=[1, 2, 3], generic_list_field=["hello"])

    lit = await TypeEngine.to_literal(primitive, PrimitiveDC, literal_type)
    print(lit)
    recovered_primitive = await TypeEngine.to_python_value(lit, PrimitiveDC)

    assert recovered_primitive.int_field == 42
    assert recovered_primitive.float_field == 3.14
    assert recovered_primitive.str_field == "hello"
    assert recovered_primitive.bool_field is None
    assert recovered_primitive.list_field == [1, 2, 3]
    assert recovered_primitive.generic_list_field == ["hello"]


@pytest.mark.asyncio
async def test_basic_dc_in_task(local_dummy_file, local_dummy_directory):
    env = flyte.TaskEnvironment(name="test-delayed-dataclass-transformer")

    @env.task
    async def t1(dc: PrimitiveDC) -> PrimitiveDC:
        return dc

    flyte.init()
    primitive = PrimitiveDC(int_field=42, float_field=3.14, list_field=[1, 2, 3], generic_list_field=["hello"])
    result = flyte.with_runcontext(mode="local").run(t1, dc=primitive).outputs()[0]

    assert result.int_field == 42
    assert result.float_field == 3.14
    assert result.str_field == "hello"
    assert result.bool_field is None
    assert result.list_field == [1, 2, 3]
    assert result.generic_list_field == ["hello"]


def test_get_signature():
    from flyte.models import NativeInterface

    async def my_func(dc: PrimitiveDC) -> Tuple[File, File, Dir, Dir]: ...

    native_interface = NativeInterface.from_callable(my_func)
    assert len(native_interface.outputs) == 4

    async def my_func_2() -> Tuple[File, File, Dir, Dir]: ...

    native_interface = NativeInterface.from_callable(my_func_2)
    assert len(native_interface.outputs) == 4

    async def my_func_3(in1: DoesNotExist) -> Tuple[File, File, Dir, Dir]:  # noqa: F821
        ...

    with pytest.raises(NameError):
        NativeInterface.from_callable(my_func_3)


@dataclass
class InnerDC:
    file: File
    dir: Dir


@dataclass
class DC:
    file: File
    dir: Dir
    inner_dc: InnerDC


@pytest.mark.asyncio
async def test_flytetypes_in_dataclass_wf(ctx_with_test_raw_data_path, local_dummy_file, local_dummy_directory):
    flyte.init(storage=S3.for_sandbox())
    env = flyte.TaskEnvironment(name="test-delayed-dataclass-transformer-flyte-types")

    @env.task
    async def t1(path: File) -> File:
        return path

    @env.task
    async def t2(path: Dir) -> Dir:
        return path

    @env.task
    async def main(dc: DC) -> Tuple[File, File, Dir, Dir]:
        return (
            await t1(path=dc.file),
            await t1(path=dc.inner_dc.file),
            await t2(path=dc.dir),
            await t2(path=dc.inner_dc.dir),
        )

    dc = DC(
        file=File(path=local_dummy_file),
        dir=Dir(path=local_dummy_directory),
        inner_dc=InnerDC(
            file=File(path=local_dummy_file),
            dir=Dir(path=local_dummy_directory),
        ),
    )
    o = flyte.run(main, dc=dc)
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
