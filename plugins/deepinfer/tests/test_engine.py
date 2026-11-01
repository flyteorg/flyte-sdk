"""Tests for `DeepInfer`, run against the stand-in DeepInfer in `tests/stub`.

Engines run in real `spawn`ed processes, exactly as on a GPU node; only the DeepInfer
engine itself is the stub.
"""

import re
from pathlib import Path
from typing import Literal, Optional

import pytest
from pydantic import BaseModel

from flyteplugins.deepinfer import DeepInfer

STUB = Path(__file__).parent / "stub"


@pytest.fixture(autouse=True)
def stub_deepinfer(monkeypatch):
    # sys.path is handed to spawned children, so engines import the stub too.
    monkeypatch.syspath_prepend(str(STUB))
    for var in ("RANK", "WORLD_SIZE", "CUDA_VISIBLE_DEVICES", "STUB_FAIL_RANK", "DEEPINFER_USE_NEXTN_DECODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("STUB_NUM_GPUS", "4")


@pytest.fixture
def model(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    return str(tmp_path)


def _field(text: str, name: str) -> str:
    return re.search(rf"{name}=(\S+)", text).group(1)


def test_results_in_input_order_one_engine_per_gpu(model):
    prompts = [f"p{i}" for i in range(10)]
    with DeepInfer(model) as llm:
        out = llm.generate(prompts, max_tokens=32)
    assert [o.rsplit("| ", 1)[1] for o in out] == prompts
    # Prompt i is served by engine i % 4, pinned to device i % 4.
    assert [_field(o, "rank") for o in out] == [str(i % 4) for i in range(10)]
    assert [_field(o, "devices") for o in out] == [str(i % 4) for i in range(10)]
    assert {_field(o, "ep") for o in out} == {"4"}
    assert {_field(o, "dp") for o in out} == {"4"}
    assert {_field(o, "max_tokens") for o in out} == {"32"}


def test_engines_stay_up_between_calls(model):
    with DeepInfer(model) as llm:
        first = {_field(o, "pid") for o in llm.generate(["a", "b", "c", "d"])}
        second = {_field(o, "pid") for o in llm.generate(["e", "f", "g", "h"])}
    assert len(first) == 4
    assert first == second


@pytest.mark.asyncio
async def test_async_call(model):
    with DeepInfer(model) as llm:
        out = await llm.generate.aio(["hello"], temperature=0)
    assert _field(out[0], "temperature") == "0"


def test_fewer_prompts_than_engines(model):
    with DeepInfer(model) as llm:
        out = llm.generate(["only"])
    assert len(out) == 1
    assert out[0].endswith("| only")


def test_empty_prompts_do_not_start_engines(model):
    llm = DeepInfer(model)
    assert llm.generate([]) == []
    assert llm._pool is None


def test_cp_size_groups_devices_per_engine(model):
    with DeepInfer(model, cp_size=2) as llm:
        out = llm.generate(["a", "b"])
    assert [_field(o, "devices") for o in out] == ["0,1", "2,3"]
    assert {_field(o, "dp") for o in out} == {"2"}


def test_env_reaches_engines_and_wins_over_defaults(model):
    with DeepInfer(model) as llm:
        default = llm.generate(["__env__ DEEPINFER_USE_NEXTN_DECODE"])
    with DeepInfer(model, env={"DEEPINFER_USE_NEXTN_DECODE": "0"}) as llm:
        overridden = llm.generate(["__env__ DEEPINFER_USE_NEXTN_DECODE"])
    assert default == ["5"]
    assert overridden == ["0"]


class Ticket(BaseModel):
    category: Literal["billing", "account"]
    urgency: Literal["low", "high"]
    summary: str
    order_id: Optional[str] = None


def test_json_schema_pydantic_returns_models(model):
    with DeepInfer(model) as llm:
        out = llm.generate(["a", "b"], json_schema=Ticket)
    assert all(isinstance(t, Ticket) for t in out)
    assert out[0].category == "billing"


def test_json_schema_dict_returns_parsed_json(model):
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
    with DeepInfer(model) as llm:
        assert llm.generate(["a"], json_schema=schema) == [{"count": 1}]


def test_regex_and_grammar(model):
    with DeepInfer(model) as llm:
        assert llm.generate(["a"], regex=r"[A-Z]{2}-[0-9]{3}") == ["AB-123"]
        assert llm.generate(["a", "b"], regex=r"(positive|negative|neutral)") == ["positive", "positive"]
        assert llm.generate(["a"], grammar='root ::= "answer=" ("yes" | "no")') == ["answer=yes;level=1"]


def test_conflicting_constraints_rejected(model):
    llm = DeepInfer(model)
    with pytest.raises(ValueError, match="at most one"):
        llm.generate(["a"], regex="x", grammar="y")


def test_engine_load_failure_is_reported(model, monkeypatch):
    monkeypatch.setenv("STUB_FAIL_RANK", "2")
    llm = DeepInfer(model)
    with pytest.raises(RuntimeError, match="engine 2 failed"):
        llm.generate(["a"])
    assert llm._pool is None


def test_multinode_fills_only_this_nodes_prompts(model, monkeypatch):
    # Node 1 of 2, 2 GPUs per node: 4 engines in total, ranks 2 and 3 live here.
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("STUB_NUM_GPUS", "2")
    with DeepInfer(model) as llm:
        out = llm.generate([f"p{i}" for i in range(6)])
    assert [o is None for o in out] == [True, True, False, False, True, True]
    assert [_field(o, "rank") for o in out if o] == ["2", "3"]
    assert {_field(o, "ep") for o in out if o} == {"4"}


class FakeDir:
    """Quacks like `flyte.io.Dir`: async and sync download to a local path."""

    def __init__(self, path):
        self.path = path
        self.downloads = 0

    async def download(self):
        self.downloads += 1
        return self.path

    def download_sync(self):
        self.downloads += 1
        return self.path


def test_model_from_flyte_dir_sync(model):
    d = FakeDir(model)
    with DeepInfer(d) as llm:
        assert len(llm.generate(["a"])) == 1
        assert len(llm.generate(["b"])) == 1
    assert d.downloads == 1  # only before the engines start


@pytest.mark.asyncio
async def test_model_from_flyte_dir_async(model):
    d = FakeDir(model)
    with DeepInfer(d) as llm:
        assert len(await llm.generate.aio(["a"])) == 1
        assert len(await llm.generate.aio(["b"])) == 1
    assert d.downloads == 1


@pytest.mark.asyncio
async def test_sync_call_inside_event_loop_points_to_aio(model):
    llm = DeepInfer(model)
    with pytest.raises(RuntimeError, match=r"generate\.aio"):
        llm.generate(["a"])


def test_construction_is_cheap_and_lazy():
    llm = DeepInfer("deepseek-ai/does-not-matter")
    assert llm._pool is None
    llm.close()  # no-op before start


def test_engine_crash_during_generate_discards_pool(model, monkeypatch):
    with DeepInfer(model) as llm:
        assert len(llm.generate(["a"])) == 1
        pool = llm._pool
        # Kill one engine behind the pool's back, as a CUDA OOM or segfault would.
        pool._procs[1].terminate()
        pool._procs[1].join(timeout=5)
        with pytest.raises(RuntimeError):
            llm.generate([f"p{i}" for i in range(8)])
        # The broken pool is gone, so the next call starts fresh instead of hanging.
        assert llm._pool is None
        assert len(llm.generate(["b"])) == 1
