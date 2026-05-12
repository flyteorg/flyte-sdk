import tempfile

import pytest
from pydantic import BaseModel, Field


@pytest.mark.asyncio
async def test_dataclass_in_pydantic_basemodel():
    from dataclasses import dataclass

    import flyte

    temp_dir = tempfile.mkdtemp()
    flyte.init(root_dir=temp_dir)
    env = flyte.TaskEnvironment(name="pydantic-dataclass-in-pydantic-basemodel")

    @dataclass
    class InnerBM:
        a: int = -1
        b: float = 3.14
        c: str = "Hello, Flyte"
        d: bool = False

    class BM(BaseModel):
        a: int = -1
        b: float = 3.14
        c: str = "Hello, Flyte"
        d: bool = False
        inner_bm: InnerBM = Field(default_factory=InnerBM)

    @env.task
    async def t_bm(bm: BM):
        assert isinstance(bm, BM)
        assert isinstance(bm.inner_bm, InnerBM)

    @env.task
    async def t_inner(inner_bm: InnerBM):
        assert isinstance(inner_bm, InnerBM)

    @env.task
    async def t_test_primitive_attributes(a: int, b: float, c: str, d: bool):
        assert isinstance(a, int), f"a is not int, it's {type(a)}"
        assert a == -1
        assert isinstance(b, float), f"b is not float, it's {type(b)}"
        assert b == 3.14
        assert isinstance(c, str), f"c is not str, it's {type(c)}"
        assert c == "Hello, Flyte"
        assert isinstance(d, bool), f"d is not bool, it's {type(d)}"
        assert d is False
        print("All primitive attributes passed strict type checks.")

    @env.task
    async def main(bm: BM):
        await t_bm(bm=bm)
        await t_inner(inner_bm=bm.inner_bm)
        await t_test_primitive_attributes(a=bm.a, b=bm.b, c=bm.c, d=bm.d)
        await t_test_primitive_attributes(a=bm.inner_bm.a, b=bm.inner_bm.b, c=bm.inner_bm.c, d=bm.inner_bm.d)

    flyte.run(main, bm=BM())


@pytest.mark.asyncio
async def test_pydantic_dataclass_in_pydantic_basemodel():
    from pydantic.dataclasses import dataclass

    import flyte

    temp_dir = tempfile.mkdtemp()
    flyte.init(root_dir=temp_dir)
    env = flyte.TaskEnvironment(name="pydantic-dataclass-in-pydantic-basemodel")

    @dataclass
    class InnerBM:
        a: int = -1
        b: float = 3.14
        c: str = "Hello, Flyte"
        d: bool = False

    class BM(BaseModel):
        a: int = -1
        b: float = 3.14
        c: str = "Hello, Flyte"
        d: bool = False
        inner_bm: InnerBM = Field(default_factory=InnerBM)

    @env.task
    async def t_bm(bm: BM):
        assert isinstance(bm, BM)
        assert isinstance(bm.inner_bm, InnerBM)

    @env.task
    async def t_inner(inner_bm: InnerBM):
        assert isinstance(inner_bm, InnerBM)

    @env.task
    async def t_test_primitive_attributes(a: int, b: float, c: str, d: bool):
        assert isinstance(a, int), f"a is not int, it's {type(a)}"
        assert a == -1
        assert isinstance(b, float), f"b is not float, it's {type(b)}"
        assert b == 3.14
        assert isinstance(c, str), f"c is not str, it's {type(c)}"
        assert c == "Hello, Flyte"
        assert isinstance(d, bool), f"d is not bool, it's {type(d)}"
        assert d is False
        print("All primitive attributes passed strict type checks.")

    @env.task
    async def main(bm: BM):
        await t_bm(bm=bm)
        await t_inner(inner_bm=bm.inner_bm)
        await t_test_primitive_attributes(a=bm.a, b=bm.b, c=bm.c, d=bm.d)
        await t_test_primitive_attributes(a=bm.inner_bm.a, b=bm.inner_bm.b, c=bm.inner_bm.c, d=bm.inner_bm.d)

    flyte.run(main, bm=BM())
