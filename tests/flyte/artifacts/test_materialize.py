"""`flyte.materialize`: a thin, lazily-imported delegator to the Union lineage planner."""

import builtins
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import flyte
import flyte.artifacts as artifacts
from flyte.errors import MaterializeError, RuntimeUserError
from flyte.io import DataFrame

features = artifacts.Artifact("features", type=DataFrame, partitions={"date": artifacts.Daily}, project="ml")


def test_materialize_without_plugin_raises():
    real_import = builtins.__import__

    def no_plugin(name, *args, **kwargs):
        if name.startswith("flyteplugins.union.factory.derived"):
            raise ModuleNotFoundError(f"No module named {name!r}", name="flyteplugins.union.factory.derived")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", no_plugin):
        with pytest.raises(MaterializeError) as exc:
            flyte.materialize(features, date=datetime(2026, 9, 8))
    assert str(exc.value).startswith("flyte.materialize requires flyteplugins-union")
    assert isinstance(exc.value, RuntimeUserError) and exc.value.code == "MaterializeError"


def test_materialize_delegates_all_arguments():
    fake = MagicMock(return_value="RUN", spec=lambda *a, **k: None)  # spec: no `.aio`, a plain callable
    mod = type(sys)("flyteplugins.union.factory.derived")
    mod.materialize = fake  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"flyteplugins.union.factory.derived": mod}):
        rng = flyte.TimeRange("2026-08-01", "2026-08-31")
        run = flyte.materialize(
            features,
            inputs={"ml.train.lr": 1e-4},
            concurrency=50,
            plan_only=True,
            rebuild=["x"],
            rebuild_all=False,
            project="p",
            domain="d",
            date=rng,
        )
    assert run == "RUN"
    fake.assert_called_once_with(
        features,
        inputs={"ml.train.lr": 1e-4},
        concurrency=50,
        plan_only=True,
        rebuild=["x"],
        rebuild_all=False,
        project="p",
        domain="d",
        date=rng,
    )


def test_materialize_surfaces_plugin_internal_import_errors():
    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name.startswith("flyteplugins.union.factory.derived"):
            raise ModuleNotFoundError("No module named 'networkx'", name="networkx")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", broken), pytest.raises(ModuleNotFoundError, match="networkx"):
        flyte.materialize("features")


def _fake_plugin(fn):
    import sys

    mod = type(sys)("flyteplugins.union.factory.derived")
    mod.materialize = fn  # type: ignore[attr-defined]
    return patch.dict(sys.modules, {"flyteplugins.union.factory.derived": mod})


def test_materialize_partitions_escape_hatch_and_queue():
    calls = []

    def plugin(target, **kwargs):
        calls.append(kwargs)
        return "RUN"

    with _fake_plugin(plugin):
        # A dimension named "project" collides with the keyword: it goes through partitions=.
        assert flyte.materialize("t", partitions={"project": "alpha", "date": 1}, date=2, queue="q") == "RUN"
        flyte.materialize("t", partitions={"region": "us"}, date=3)
        flyte.materialize("t")
    first, second, third = calls
    assert first["project"] is None and first["queue"] == "q"
    assert first["partitions"] == {"project": "alpha", "date": 2}  # keyword wins over partitions=
    assert second["region"] == "us" and second["date"] == 3 and "partitions" not in second  # spread as before
    assert "queue" not in third and "partitions" not in third


@pytest.mark.asyncio
async def test_materialize_aio_prefers_plugin_aio():
    class Plugin:
        def __call__(self, *a, **k):
            raise AssertionError("sync path must not be used from .aio")

        async def aio(self, target, **kwargs):
            return ("ASYNC", target, kwargs["concurrency"])

    with _fake_plugin(Plugin()):
        assert await flyte.materialize.aio("t", concurrency=5) == ("ASYNC", "t", 5)


@pytest.mark.asyncio
async def test_materialize_aio_runs_sync_plugin_off_loop():
    with _fake_plugin(lambda target, **kw: ("SYNC", target)):
        assert await flyte.materialize.aio("t") == ("SYNC", "t")


def test_materialize_forwards_source_check_only_when_disabled():
    calls = []

    def plugin(target, **kwargs):
        calls.append(kwargs)
        return "RUN"

    with _fake_plugin(plugin):
        flyte.materialize("t", date=1, source_check=False)
        flyte.materialize("t", date=1)
        # A dimension named like the keyword still reaches the planner, through partitions=.
        flyte.materialize("t", partitions={"source_check": "x"})
    disabled, default, dim = calls
    assert disabled["source_check"] is False and disabled["date"] == 1 and "partitions" not in disabled
    assert "source_check" not in default  # an older plugin without the keyword keeps working
    assert dim["partitions"] == {"source_check": "x"}
