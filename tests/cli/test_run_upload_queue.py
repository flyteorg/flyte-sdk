"""`flyte run --queue` uploads local File/Dir inputs to the storage of the run's queue.

Click converts (and so uploads) those inputs while it parses the command line, before the run is
submitted, so the run command sets the upload queue around parsing. The upload helper is patched at
its async core, which runs on the syncify loop thread, so these tests also prove the queue reaches
the upload across that thread.
"""

import threading
from pathlib import Path

import pytest

import flyte
import flyte.remote._data as data
from flyte._upload_queue import current_upload_queue
from flyte.cli._run import RunArguments, RunRemoteTaskCommand, RunTaskCommand
from flyte.io import Dir, File


@pytest.fixture
def uploads(monkeypatch):
    seen = []

    async def fake_upload_single_file(cfg, fp, verify=True, basedir=None, **kwargs):
        seen.append((Path(fp).name, current_upload_queue(), threading.current_thread() is threading.main_thread()))
        prefix = basedir or "p"
        return "md5", f"s3://bucket/{prefix}/{Path(fp).name}"

    monkeypatch.setattr(data, "_upload_single_file", fake_upload_single_file)
    monkeypatch.setattr(data, "ensure_client", lambda: None)
    monkeypatch.setattr(data, "get_init_config", lambda: None)
    return seen


@pytest.fixture
def inputs(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_text("a")
    f = tmp_path / "model.bin"
    f.write_text("m")
    return d, f


def _task(queue=None):
    env = flyte.TaskEnvironment(name="upload_queue_test", queue=queue)

    @env.task
    async def train(data: Dir, model: File) -> str:
        return ""

    return train


@pytest.mark.parametrize(
    "cli_queue, task_queue, want",
    [
        ("other-cloud", None, "other-cloud"),  # --queue
        (None, "other-cloud", "other-cloud"),  # the task's own queue
        ("gpu", "other-cloud", "gpu"),  # --queue overrides the task's
        (None, None, None),  # neither: the project/domain default, as before
    ],
)
def test_local_task_inputs_upload_to_the_run_queue(uploads, inputs, cli_queue, task_queue, want):
    d, f = inputs
    cmd = RunTaskCommand(obj_name="train", obj=_task(task_queue), run_args=RunArguments(queue=cli_queue))
    ctx = cmd.make_context("train", ["--data", str(d), "--model", str(f)])

    assert isinstance(ctx.params["data"], Dir) and isinstance(ctx.params["model"], File)
    assert sorted(uploads) == [("a.txt", want, False), ("model.bin", want, False)]
    assert current_upload_queue() is None, "the upload queue does not leak past parsing"


def test_deployed_task_inputs_upload_to_the_cli_queue(uploads, inputs, monkeypatch):
    d, f = inputs
    cmd = RunRemoteTaskCommand(task_name="t", run_args=RunArguments(queue="other-cloud"), version="v1", name="t")
    # The deployed task's inputs, as get_params builds them from its interface.
    monkeypatch.setattr(
        RunRemoteTaskCommand,
        "get_params",
        lambda self, ctx: RunTaskCommand(obj_name="train", obj=_task(), run_args=RunArguments()).get_params(ctx),
    )
    cmd.make_context("t", ["--data", str(d), "--model", str(f)])

    assert sorted(uploads) == [("a.txt", "other-cloud", False), ("model.bin", "other-cloud", False)]
