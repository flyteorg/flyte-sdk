"""Warm-up runs: `with_runcontext(warmup=True)` / `Trigger(warmup=True)` set the warm-up run label."""

import mock
import pytest
from flyteidl2.core import interface_pb2
from mock.mock import AsyncMock

import flyte
from flyte._internal.runtime.trigger_serde import to_task_trigger
from flyte._warmup import WARMUP_LABEL, WARMUP_LABEL_VALUE, with_warmup_label

from .test_run_runspec_chars import _patch_build, _run_and_capture, task1


def test_with_warmup_label():
    assert with_warmup_label(None, False) is None
    assert with_warmup_label({"a": "b"}, False) == {"a": "b"}
    assert with_warmup_label(None, True) == {WARMUP_LABEL: WARMUP_LABEL_VALUE}
    labels = {"team": "ml"}
    assert with_warmup_label(labels, True) == {"team": "ml", WARMUP_LABEL: WARMUP_LABEL_VALUE}
    assert labels == {"team": "ml"}, "caller's dict is not mutated"


def test_label_contract():
    # The backend (leaseworker + billing) matches exactly this key and value.
    assert WARMUP_LABEL == "flyte.org/warmup"
    assert WARMUP_LABEL_VALUE == "true"


@pytest.mark.asyncio
@_patch_build
async def test_runcontext_warmup_sets_run_label(mock_code_bundler, mock_build_image_bg):
    req = await _run_and_capture(mock_build_image_bg, mock_code_bundler, warmup=True, labels={"team": "ml"})
    assert req.run_spec.labels.values[WARMUP_LABEL] == WARMUP_LABEL_VALUE
    assert req.run_spec.labels.values["team"] == "ml"


@pytest.mark.asyncio
@_patch_build
async def test_runcontext_default_has_no_warmup_label(mock_code_bundler, mock_build_image_bg):
    req = await _run_and_capture(mock_build_image_bg, mock_code_bundler, labels={"team": "ml"})
    assert WARMUP_LABEL not in req.run_spec.labels.values


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local", "hybrid"])
async def test_runcontext_warmup_rejected_outside_remote(mode):
    from flyte._run import _Runner

    runner = flyte.with_runcontext(mode=mode, warmup=True, name="r", run_base_dir="s3://b/md")
    with mock.patch.object(_Runner, "_run_local", new_callable=AsyncMock) as local:
        with mock.patch.object(_Runner, "_run_hybrid", new_callable=AsyncMock) as hybrid:
            with pytest.raises(ValueError, match="warmup"):
                await runner.run.aio(task1, "hello")
            local.assert_not_called()
            hybrid.assert_not_called()


@pytest.mark.asyncio
async def test_trigger_warmup_sets_run_label():
    trigger = flyte.Trigger(
        name="keep_warm",
        automation=flyte.FixedRate(1),
        labels={"team": "ml"},
        warmup=True,
    )
    result = await to_task_trigger(trigger, "test_task", interface_pb2.VariableMap(), [])
    assert result.spec.run_spec.labels.values == {"team": "ml", WARMUP_LABEL: WARMUP_LABEL_VALUE}


@pytest.mark.asyncio
async def test_trigger_default_has_no_warmup_label():
    trigger = flyte.Trigger(name="t", automation=flyte.FixedRate(1))
    result = await to_task_trigger(trigger, "test_task", interface_pb2.VariableMap(), [])
    assert WARMUP_LABEL not in result.spec.run_spec.labels.values


def test_trigger_constructors_accept_warmup():
    for ctor in (
        flyte.Trigger.minutely,
        flyte.Trigger.hourly,
        flyte.Trigger.daily,
        flyte.Trigger.weekly,
        flyte.Trigger.monthly,
    ):
        assert ctor(warmup=True).warmup is True
        assert ctor().warmup is False
