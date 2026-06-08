"""
Passing plain dicts (and partial dicts) to BaseModel / dataclass task inputs.

A client wants to launch a task whose input is a Pydantic ``BaseModel`` (or dataclass) *without*
constructing the model object — and ideally without even importing the model class, because the
task definition and the calling code live in different repos.

``flyte.run`` coerces a dict into the target BaseModel/dataclass at the type-engine layer:

* full dict     -> the model is built from the dict
* partial dict  -> omitted fields are filled from the model's field defaults
* it also works for remote tasks fetched via ``Task.get()``, where the client does not have the
  class at all (the input type is reconstructed from the task's schema, defaults included)

Note the ``tags: dict[str, str] | None`` field below — that exact shape used to crash with
``KeyError: 'title'`` when the type was reconstructed on the client side.

Run locally to see it work:

    python examples/basics/types/dict_to_model_inputs.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel

import flyte

env = flyte.TaskEnvironment(name="dict_to_model_inputs")


class ParametersOption(str, Enum):
    DAL_OMS = "dal_oms"
    DAL_ONLY = "dal_only"


class QueryInfo(BaseModel):
    """A Pydantic input model."""

    query: str  # required, no default
    name: str = "default"  # defaulted field -- can be omitted from the dict
    tags: dict[str, str] | None = None  # the field shape from the original bug report


@env.task
async def regen_in_batch(
    query_info: QueryInfo,
    parameters_option: ParametersOption = ParametersOption.DAL_OMS,
    min_success_ratio: float = 0.9,
    concurrency: int = 4,
) -> str:
    return (
        f"query={query_info.query!r} name={query_info.name!r} tags={query_info.tags} "
        f"option={parameters_option.value} ratio={min_success_ratio} concurrency={concurrency}"
    )


# A dataclass input works exactly the same way.
@dataclass
class QueryInfoDC:
    query: str
    name: str = "default"
    tags: Optional[dict[str, str]] = None


@env.task
async def regen_in_batch_dc(query_info: QueryInfoDC) -> str:
    return f"query={query_info.query!r} name={query_info.name!r} tags={query_info.tags}"


if __name__ == "__main__":
    # Local execution, so you can run this file directly and verify the coercion.
    flyte.init()

    # 1) Full dict -- no need to construct QueryInfo(...).
    r1 = flyte.run(regen_in_batch, query_info={"query": "select 1", "name": "full", "tags": {"team": "ml"}})
    print("full dict   :", r1.outputs())

    # 2) Minimal dict -- name/tags omitted, filled from QueryInfo's field defaults.
    r2 = flyte.run(regen_in_batch, query_info={"query": "select 2"})
    print("partial dict:", r2.outputs())

    # 3) Partial dict + override a task-level defaulted argument.
    r3 = flyte.run(
        regen_in_batch,
        query_info={"query": "select 3", "tags": {"k": "v"}},
        parameters_option=ParametersOption.DAL_ONLY,
    )
    print("override arg:", r3.outputs())

    # 4) Same behavior for a dataclass input.
    r4 = flyte.run(regen_in_batch_dc, query_info={"query": "select 4"})
    print("dataclass   :", r4.outputs())

    # ---- Decoupled / remote usage (no QueryInfo import needed) ----------------------------------
    # In a separate repo, a client can fetch the deployed task and launch it with a plain dict --
    # it does NOT need to import QueryInfo. The input type is reconstructed from the task's schema
    # (with defaults), so a minimal dict still works:
    #
    #     import flyte
    #     from flyte.remote import Task
    #
    #     flyte.init_from_config()
    #     task = Task.get("regen_in_batch", project="...", domain="...", auto_version="latest")
    #     run = flyte.with_runcontext(mode="remote").run(task, query_info={"query": "select 1"})
    #     run.wait()
