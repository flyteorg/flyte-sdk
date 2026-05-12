import asyncio
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

import flyte

env = flyte.TaskEnvironment(name="inputs_pydantic_nested_types")


class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class Inner(BaseModel):
    name: str
    value: int


class ModelWithEnum(BaseModel):
    label: str
    status: Status


@env.task
async def nested_lists(matrix: List[List[int]]) -> str:
    return f"Matrix {len(matrix)}x{len(matrix[0])}: {matrix}"


@env.task
async def list_of_dicts(records: List[Dict[str, int]]) -> str:
    return f"Records ({len(records)} entries): {records}"


@env.task
async def dict_of_dicts(nested_map: Dict[str, Dict[str, int]]) -> str:
    return f"Nested map keys: {list(nested_map.keys())}, values: {nested_map}"


@env.task
async def nested_models(items: List[List[Inner]]) -> str:
    names = [[m.name for m in row] for row in items]
    return f"Nested model names: {names}"


@env.task
async def dict_of_models(models: Dict[str, Inner]) -> str:
    return f"Dict of models: {list(models.keys())}"


@env.task
async def enum_in_models(jobs: List[ModelWithEnum]) -> str:
    return f"Jobs: {[(j.label, j.status.value) for j in jobs]}"


@env.task
async def optional_model(inner: Optional[Inner] = None) -> str:
    return f"Optional inner: {inner}"


class ComplexNestedModel(BaseModel):
    nested_list: List[List[Inner]]
    dict_of_model_lists: Dict[str, List[Inner]]
    list_of_model_dicts: List[Dict[str, Inner]]
    enum_model_map: Dict[str, ModelWithEnum]
    list_of_dicts: List[Dict[str, int]]
    optional_inner: Optional[Inner] = None


@env.task
async def complex_nesting(data: ComplexNestedModel) -> str:
    result = f"Nested list rows: {len(data.nested_list)}"
    result += f"\nDict of model lists keys: {list(data.dict_of_model_lists.keys())}"
    result += f"\nList of model dicts count: {len(data.list_of_model_dicts)}"
    result += f"\nEnum model map: {[(k, v.status.value) for k, v in data.enum_model_map.items()]}"
    result += f"\nList of dicts: {data.list_of_dicts}"
    result += f"\nOptional inner: {data.optional_inner}"
    return result


@env.task
async def main() -> str:
    r1, r2, r3, r4, r5, r6, r7, r8, r9 = await asyncio.gather(
        nested_lists(matrix=[[1, 2], [3, 4]]),
        list_of_dicts(records=[{"a": 1, "b": 2}, {"c": 3}]),
        dict_of_dicts(nested_map={"outer": {"inner_key": 10}}),
        nested_models(items=[[Inner(name="a", value=1)], [Inner(name="b", value=2)]]),
        dict_of_models(models={"x": Inner(name="c", value=3)}),
        enum_in_models(
            jobs=[
                ModelWithEnum(label="job1", status=Status.ACTIVE),
                ModelWithEnum(label="job2", status=Status.INACTIVE),
            ]
        ),
        optional_model(inner=Inner(name="d", value=4)),
        optional_model(),
        complex_nesting(
            data=ComplexNestedModel(
                nested_list=[[Inner(name="a", value=1)], [Inner(name="b", value=2)]],
                dict_of_model_lists={"group1": [Inner(name="c", value=3), Inner(name="d", value=4)]},
                list_of_model_dicts=[{"x": Inner(name="e", value=5)}, {"y": Inner(name="f", value=6)}],
                enum_model_map={
                    "j1": ModelWithEnum(label="job1", status=Status.ACTIVE),
                    "j2": ModelWithEnum(label="job2", status=Status.INACTIVE),
                },
                list_of_dicts=[{"k1": 10, "k2": 20}],
                optional_inner=Inner(name="g", value=7),
            )
        ),
    )

    return f"{r1}\n{r2}\n{r3}\n{r4}\n{r5}\n{r6}\n{r7}\n{r8}\n{r9}"


if __name__ == "__main__":
    flyte.init_from_config()

    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
