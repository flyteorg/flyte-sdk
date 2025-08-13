import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import mock
import pytest
from mashumaro.mixins.json import DataClassJSONMixin
from typing_extensions import Annotated

import flyte
from flyte.io import Dir, File
from flyte.io._dataframe import DataFrame
from flyte.types._type_engine import (
    DataclassTransformer,
    TypeEngine,
)

pd = pytest.importorskip("pandas")


@pytest.fixture
def local_dummy_txt_file():
    fd, path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as tmp:
            tmp.write("Hello World")
        yield path
    finally:
        os.remove(path)


@pytest.fixture
def local_dummy_directory():
    temp_dir = tempfile.TemporaryDirectory()
    try:
        with open(os.path.join(temp_dir.name, "file"), "w") as tmp:
            tmp.write("Hello world")
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


@pytest.mark.asyncio
async def test_dataclass():
    import flyte

    env = flyte.TaskEnvironment(name="test-dc-dc")

    @dataclass
    class AppParams:
        snapshotDate: str
        region: str
        preprocess: bool
        listKeys: List[str]

    @env.task
    async def t1() -> AppParams:
        ap = AppParams(snapshotDate="4/5/2063", region="us-west-3", preprocess=False, listKeys=["a", "b"])
        return ap

    @env.task
    async def wf() -> AppParams:
        return await t1()

    res = await wf()
    assert res.region == "us-west-3"
    res = flyte.run(wf)
    assert res.outputs().region == "us-west-3"


def test_dataclass_assert_works_for_annotated():
    @dataclass
    class MyDC(DataClassJSONMixin):
        my_str: str

    d = Annotated[MyDC, "tag"]
    DataclassTransformer().assert_type(d, MyDC(my_str="hi"))


@pytest.mark.asyncio
async def test_pure_dataclasses_with_python_types():
    import flyte

    env = flyte.TaskEnvironment(name="test-dc-dc")

    @dataclass
    class DC:
        string: Optional[str] = None

    @dataclass
    class DCWithOptional:
        string: Optional[str] = None
        dc: Optional[DC] = None
        list_dc: Optional[List[DC]] = None
        list_list_dc: Optional[List[List[DC]]] = None
        dict_dc: Optional[Dict[str, DC]] = None
        dict_dict_dc: Optional[Dict[str, Dict[str, DC]]] = None
        dict_list_dc: Optional[Dict[str, List[DC]]] = None
        list_dict_dc: Optional[List[Dict[str, DC]]] = None

    @env.task
    async def t1() -> DCWithOptional:
        return DCWithOptional(
            string="a",
            dc=DC(string="b"),
            list_dc=[DC(string="c"), DC(string="d")],
            list_list_dc=[[DC(string="e"), DC(string="f")]],
            list_dict_dc=[{"g": DC(string="h"), "i": DC(string="j")}, {"k": DC(string="l"), "m": DC(string="n")}],
            dict_dc={"o": DC(string="p"), "q": DC(string="r")},
            dict_dict_dc={"s": {"t": DC(string="u"), "v": DC(string="w")}},
            dict_list_dc={"x": [DC(string="y"), DC(string="z")], "aa": [DC(string="bb"), DC(string="cc")]},
        )

    @env.task
    async def t2() -> DCWithOptional:
        return DCWithOptional()

    output = DCWithOptional(
        string="a",
        dc=DC(string="b"),
        list_dc=[DC(string="c"), DC(string="d")],
        list_list_dc=[[DC(string="e"), DC(string="f")]],
        list_dict_dc=[{"g": DC(string="h"), "i": DC(string="j")}, {"k": DC(string="l"), "m": DC(string="n")}],
        dict_dc={"o": DC(string="p"), "q": DC(string="r")},
        dict_dict_dc={"s": {"t": DC(string="u"), "v": DC(string="w")}},
        dict_list_dc={"x": [DC(string="y"), DC(string="z")], "aa": [DC(string="bb"), DC(string="cc")]},
    )

    dc1 = await t1()
    dc2 = await t2()

    assert dc1 == output
    assert dc2.string is None
    assert dc2.dc is None

    DataclassTransformer().assert_type(DCWithOptional, dc1)
    DataclassTransformer().assert_type(DCWithOptional, dc2)

    o1 = flyte.run(t1)
    dc1 = o1.outputs()

    o2 = flyte.run(t2)
    dc2 = o2.outputs()

    assert dc1 == output
    assert dc2.string is None
    assert dc2.dc is None

    DataclassTransformer().assert_type(DCWithOptional, dc1)
    DataclassTransformer().assert_type(DCWithOptional, dc2)


@pytest.mark.asyncio
async def test_pure_dataclasses_with_python_types_get_literal_type_and_to_python_value():
    @dataclass
    class DC:
        string: Optional[str] = None

    @dataclass
    class DCWithOptional:
        string: Optional[str] = None
        dc: Optional[DC] = None
        list_dc: Optional[List[DC]] = None
        list_list_dc: Optional[List[List[DC]]] = None
        dict_dc: Optional[Dict[str, DC]] = None
        dict_dict_dc: Optional[Dict[str, Dict[str, DC]]] = None
        dict_list_dc: Optional[Dict[str, List[DC]]] = None
        list_dict_dc: Optional[List[Dict[str, DC]]] = None

    o = DCWithOptional()
    lt = TypeEngine.to_literal_type(DCWithOptional)
    lv = await TypeEngine.to_literal(o, DCWithOptional, lt)
    assert lv is not None
    pv = await TypeEngine.to_python_value(lv, DCWithOptional)
    assert isinstance(pv, DCWithOptional)
    DataclassTransformer().assert_type(DCWithOptional, pv)

    o = DCWithOptional(
        string="a",
        dc=DC(string="b"),
        list_dc=[DC(string="c"), DC(string="d")],
        list_list_dc=[[DC(string="e"), DC(string="f")]],
        list_dict_dc=[{"g": DC(string="h"), "i": DC(string="j")}, {"k": DC(string="l"), "m": DC(string="n")}],
        dict_dc={"o": DC(string="p"), "q": DC(string="r")},
        dict_dict_dc={"s": {"t": DC(string="u"), "v": DC(string="w")}},
        dict_list_dc={"x": [DC(string="y"), DC(string="z")], "aa": [DC(string="bb"), DC(string="cc")]},
    )
    lt = TypeEngine.to_literal_type(DCWithOptional)
    lv = await TypeEngine.to_literal(o, DCWithOptional, lt)
    assert lv is not None
    pv = await TypeEngine.to_python_value(lv, DCWithOptional)
    assert isinstance(pv, DCWithOptional)
    DataclassTransformer().assert_type(DCWithOptional, pv)


@pytest.mark.asyncio
async def test_pure_dataclasses_with_flyte_types(local_dummy_txt_file, local_dummy_directory):
    env = flyte.TaskEnvironment(name="test-dc-transformer")
    await flyte.init.aio()

    @dataclass
    class FlyteTypes:
        flytefile: Optional[File] = None
        flytedir: Optional[Dir] = None
        _structured_dataset: Optional[DataFrame] = None

    @dataclass
    class NestedFlyteTypes:
        flytefile: Optional[File] = None
        flytedir: Optional[Dir] = None
        _structured_dataset: Optional[DataFrame] = None
        flyte_types: Optional[FlyteTypes] = None
        list_flyte_types: Optional[List[FlyteTypes]] = None
        dict_flyte_types: Optional[Dict[str, FlyteTypes]] = None
        optional_flyte_types: Optional[FlyteTypes] = None

    @env.task
    async def pass_and_return_flyte_types(nested_flyte_types: NestedFlyteTypes) -> NestedFlyteTypes:
        return nested_flyte_types

    @env.task
    async def generate_sd() -> DataFrame:
        return DataFrame.from_existing_remote("s3://my-s3-bucket/data/test_sd", format="parquet")

    @env.task
    async def create_local_dir(path: str) -> Dir:
        return await Dir.from_local(local_path=path)

    @env.task
    async def create_local_dir_by_str(path: str) -> Dir:
        return await Dir.from_local(local_path=path)

    @env.task
    async def create_local_file(path: str) -> File:
        return await File.from_local(local_path=path)

    @env.task
    async def create_local_file_with_str(path: str) -> File:
        return await File.from_local(local_path=path)

    @env.task
    async def generate_nested_flyte_types(
        local_file: File,
        local_dir: Dir,
        sd: DataFrame,
        local_file_by_str: File,
        local_dir_by_str: Dir,
    ) -> NestedFlyteTypes:
        ft = FlyteTypes(
            flytefile=local_file,
            flytedir=local_dir,
            _structured_dataset=sd,
        )

        return NestedFlyteTypes(
            flytefile=local_file,
            flytedir=local_dir,
            _structured_dataset=sd,
            flyte_types=FlyteTypes(
                flytefile=local_file_by_str,
                flytedir=local_dir_by_str,
                _structured_dataset=sd,
            ),
            list_flyte_types=[ft, ft, ft],
            dict_flyte_types={"a": ft, "b": ft, "c": ft},
        )

    @env.task
    async def nested_dc_wf(txt_path: str, dir_path: str) -> NestedFlyteTypes:
        local_file = await create_local_file(path=txt_path)
        local_dir = await create_local_dir(path=dir_path)
        local_file_by_str = await create_local_file_with_str(path=txt_path)
        local_dir_by_str = await create_local_dir_by_str(path=dir_path)
        sd = await generate_sd()
        nested_flyte_types = await generate_nested_flyte_types(
            local_file=local_file,
            local_dir=local_dir,
            local_file_by_str=local_file_by_str,
            local_dir_by_str=local_dir_by_str,
            sd=sd,
        )
        old_flyte_types = await pass_and_return_flyte_types(nested_flyte_types=nested_flyte_types)
        return await pass_and_return_flyte_types(nested_flyte_types=old_flyte_types)

    @env.task
    async def get_empty_nested_type() -> NestedFlyteTypes:
        return NestedFlyteTypes()

    @env.task
    async def empty_nested_dc_wf() -> NestedFlyteTypes:
        return await get_empty_nested_type()

    nested_flyte_types = flyte.run(nested_dc_wf, txt_path=local_dummy_txt_file, dir_path=local_dummy_directory)
    DataclassTransformer().assert_type(NestedFlyteTypes, nested_flyte_types.outputs())

    empty_nested_flyte_types = await empty_nested_dc_wf()
    DataclassTransformer().assert_type(NestedFlyteTypes, empty_nested_flyte_types)


# For mashumaro dataclasses mixins, it's equal to use @dataclasses only
@pytest.mark.asyncio
async def test_mashumaro_dataclasses_json_mixin_with_python_types():
    import flyte

    await flyte.init.aio()
    env = flyte.TaskEnvironment(name="test-dc-dc-mashu")

    @dataclass
    class DC(DataClassJSONMixin):
        string: Optional[str] = None

    @dataclass
    class DCWithOptional(DataClassJSONMixin):
        string: Optional[str] = None
        dc: Optional[DC] = None
        list_dc: Optional[List[DC]] = None
        list_list_dc: Optional[List[List[DC]]] = None
        dict_dc: Optional[Dict[str, DC]] = None
        dict_dict_dc: Optional[Dict[str, Dict[str, DC]]] = None
        dict_list_dc: Optional[Dict[str, List[DC]]] = None
        list_dict_dc: Optional[List[Dict[str, DC]]] = None

    @env.task
    async def t1() -> DCWithOptional:
        return DCWithOptional(
            string="a",
            dc=DC(string="b"),
            list_dc=[DC(string="c"), DC(string="d")],
            list_list_dc=[[DC(string="e"), DC(string="f")]],
            list_dict_dc=[{"g": DC(string="h"), "i": DC(string="j")}, {"k": DC(string="l"), "m": DC(string="n")}],
            dict_dc={"o": DC(string="p"), "q": DC(string="r")},
            dict_dict_dc={"s": {"t": DC(string="u"), "v": DC(string="w")}},
            dict_list_dc={"x": [DC(string="y"), DC(string="z")], "aa": [DC(string="bb"), DC(string="cc")]},
        )

    @env.task
    async def t2() -> DCWithOptional:
        return DCWithOptional()

    output = DCWithOptional(
        string="a",
        dc=DC(string="b"),
        list_dc=[DC(string="c"), DC(string="d")],
        list_list_dc=[[DC(string="e"), DC(string="f")]],
        list_dict_dc=[{"g": DC(string="h"), "i": DC(string="j")}, {"k": DC(string="l"), "m": DC(string="n")}],
        dict_dc={"o": DC(string="p"), "q": DC(string="r")},
        dict_dict_dc={"s": {"t": DC(string="u"), "v": DC(string="w")}},
        dict_list_dc={"x": [DC(string="y"), DC(string="z")], "aa": [DC(string="bb"), DC(string="cc")]},
    )

    o1 = flyte.run(t1)
    dc1 = o1.outputs()
    o2 = flyte.run(t2)
    dc2 = o2.outputs()

    assert dc1 == output
    assert dc2.string is None
    assert dc2.dc is None

    DataclassTransformer().assert_type(DCWithOptional, dc1)
    DataclassTransformer().assert_type(DCWithOptional, dc2)


@pytest.mark.asyncio
async def test_ret_unions():
    import flyte

    await flyte.init.aio()
    env = flyte.TaskEnvironment(name="test-dc-ret-flyte")

    @dataclass
    class DC:
        my_string: str

    @dataclass
    class DCWithOptional:
        my_float: float

    @env.task
    async def make_union(a: int) -> Union[DC, DCWithOptional]:
        if a > 10:
            return DC(my_string="hello")
        else:
            return DCWithOptional(my_float=3.14)

    @env.task
    async def make_union_wf(a: int) -> Union[DC, DCWithOptional]:
        return await make_union(a=a)

    dc = await make_union_wf(a=15)
    assert dc.my_string == "hello"
    dc = await make_union_wf(a=5)
    assert dc.my_float == 3.14

    o = flyte.run(make_union_wf, a=15)
    dc = o.outputs()
    assert dc.my_string == "hello"
    o = flyte.run(make_union_wf, a=5)
    dc = o.outputs()
    assert dc.my_float == 3.14


@pytest.mark.asyncio
async def test_mashumaro_dataclasses_json_mixin_with_python_types_get_literal_type_and_to_python_value():
    @dataclass
    class DC(DataClassJSONMixin):
        string: Optional[str] = None

    @dataclass
    class DCWithOptional(DataClassJSONMixin):
        string: Optional[str] = None
        dc: Optional[DC] = None
        list_dc: Optional[List[DC]] = None
        list_list_dc: Optional[List[List[DC]]] = None
        dict_dc: Optional[Dict[str, DC]] = None
        dict_dict_dc: Optional[Dict[str, Dict[str, DC]]] = None
        dict_list_dc: Optional[Dict[str, List[DC]]] = None
        list_dict_dc: Optional[List[Dict[str, DC]]] = None

    o = DCWithOptional()
    lt = TypeEngine.to_literal_type(DCWithOptional)
    lv = await TypeEngine.to_literal(o, DCWithOptional, lt)
    assert lv is not None
    pv = await TypeEngine.to_python_value(lv, DCWithOptional)
    assert isinstance(pv, DCWithOptional)
    DataclassTransformer().assert_type(DCWithOptional, pv)

    o = DCWithOptional(
        string="a",
        dc=DC(string="b"),
        list_dc=[DC(string="c"), DC(string="d")],
        list_list_dc=[[DC(string="e"), DC(string="f")]],
        list_dict_dc=[{"g": DC(string="h"), "i": DC(string="j")}, {"k": DC(string="l"), "m": DC(string="n")}],
        dict_dc={"o": DC(string="p"), "q": DC(string="r")},
        dict_dict_dc={"s": {"t": DC(string="u"), "v": DC(string="w")}},
        dict_list_dc={"x": [DC(string="y"), DC(string="z")], "aa": [DC(string="bb"), DC(string="cc")]},
    )
    lt = TypeEngine.to_literal_type(DCWithOptional)
    lv = await TypeEngine.to_literal(o, DCWithOptional, lt)
    assert lv is not None
    pv = await TypeEngine.to_python_value(lv, DCWithOptional)
    assert isinstance(pv, DCWithOptional)
    DataclassTransformer().assert_type(DCWithOptional, pv)


@pytest.mark.asyncio
async def test_dataclass_union_primitive_types_and_enum():
    import flyte

    await flyte.init.aio()
    env = flyte.TaskEnvironment(name="test-dc-flyte-enum")

    class Status(Enum):
        PENDING = "pending"
        APPROVED = "approved"
        REJECTED = "rejected"

    @dataclass
    class DC:
        grid: Dict[str, List[Optional[Union[int, str, Status, float, bool]]]] = field(
            default_factory=lambda: {
                "all_types": [None, "sqrt", Status.PENDING, 1, -1, 0, -1.0, True, False],
            }
        )

    @env.task
    async def my_task(dc: DC) -> DC:
        return dc

    await my_task(dc=DC())
    flyte.run(my_task, dc=DC())


@pytest.mark.asyncio
async def test_frozen_dataclass():
    import flyte

    env = flyte.TaskEnvironment(name="test-dc-flyte-enum")

    @dataclass(frozen=True)
    class FrozenDataclass:
        a: int = 1
        b: float = 2.0
        c: bool = True
        d: str = "hello"

    @env.task
    async def t1(dc: FrozenDataclass) -> (int, float, bool, str):
        return dc.a, dc.b, dc.c, dc.d

    a, b, c, d = await t1(dc=FrozenDataclass())
    assert a == 1
    assert b == 2.0
    assert c is True
    assert d == "hello"
    await flyte.init.aio()
    o = flyte.run(t1, dc=FrozenDataclass())
    a, b, c, d = o.outputs()
    assert a == 1
    assert b == 2.0
    assert c is True
    assert d == "hello"


@pytest.mark.asyncio
async def test_pure_frozen_dataclasses_with_python_types():
    import flyte

    env = flyte.TaskEnvironment(name="test-dc-flyte-enum-frozen")

    @dataclass(frozen=True)
    class DC:
        string: Optional[str] = None

    @dataclass(frozen=True)
    class DCWithOptional:
        string: Optional[str] = None
        dc: Optional[DC] = None
        list_dc: Optional[List[DC]] = None
        list_list_dc: Optional[List[List[DC]]] = None
        dict_dc: Optional[Dict[str, DC]] = None
        dict_dict_dc: Optional[Dict[str, Dict[str, DC]]] = None
        dict_list_dc: Optional[Dict[str, List[DC]]] = None
        list_dict_dc: Optional[List[Dict[str, DC]]] = None

    @env.task
    async def t1() -> DCWithOptional:
        return DCWithOptional(
            string="a",
            dc=DC(string="b"),
            list_dc=[DC(string="c"), DC(string="d")],
            list_list_dc=[[DC(string="e"), DC(string="f")]],
            list_dict_dc=[{"g": DC(string="h"), "i": DC(string="j")}, {"k": DC(string="l"), "m": DC(string="n")}],
            dict_dc={"o": DC(string="p"), "q": DC(string="r")},
            dict_dict_dc={"s": {"t": DC(string="u"), "v": DC(string="w")}},
            dict_list_dc={"x": [DC(string="y"), DC(string="z")], "aa": [DC(string="bb"), DC(string="cc")]},
        )

    @env.task
    async def t2() -> DCWithOptional:
        return DCWithOptional()

    output = DCWithOptional(
        string="a",
        dc=DC(string="b"),
        list_dc=[DC(string="c"), DC(string="d")],
        list_list_dc=[[DC(string="e"), DC(string="f")]],
        list_dict_dc=[{"g": DC(string="h"), "i": DC(string="j")}, {"k": DC(string="l"), "m": DC(string="n")}],
        dict_dc={"o": DC(string="p"), "q": DC(string="r")},
        dict_dict_dc={"s": {"t": DC(string="u"), "v": DC(string="w")}},
        dict_list_dc={"x": [DC(string="y"), DC(string="z")], "aa": [DC(string="bb"), DC(string="cc")]},
    )

    dc1 = await t1()
    dc2 = await t2()

    assert dc1 == output
    assert dc2.string is None
    assert dc2.dc is None

    DataclassTransformer().assert_type(DCWithOptional, dc1)
    DataclassTransformer().assert_type(DCWithOptional, dc2)

    o1 = flyte.run(t1)
    dc1 = o1.outputs()
    o2 = flyte.run(t2)
    dc2 = o2.outputs()

    assert dc1 == output
    assert dc2.string is None
    assert dc2.dc is None

    DataclassTransformer().assert_type(DCWithOptional, dc1)
    DataclassTransformer().assert_type(DCWithOptional, dc2)


@pytest.mark.asyncio
@mock.patch("flyte.storage._remote_fs.RemoteFSPathResolver")
async def test_modify_literal_uris_call(mock_resolver, ctx_with_test_raw_data_path):
    sd = DataFrame._create(val=pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    @dataclass
    class DC1:
        s: DataFrame

    bm = DC1(s=sd)

    def mock_resolve_remote_path(flyte_uri: str):
        p = Path(flyte_uri)
        if p.exists():
            return "/my/replaced/val"
        return ""

    mock_resolver.resolve_remote_path.side_effect = mock_resolve_remote_path
    mock_resolver.protocol = "/"

    lt = TypeEngine.to_literal_type(DC1)
    lit = await TypeEngine.to_literal(bm, DC1, lt)

    bm_revived = await TypeEngine.to_python_value(lit, DC1)
    assert bm_revived.s.literal.uri == "/my/replaced/val"
