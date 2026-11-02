"""print() and raw stdout/stderr lines in a reusable container carry their run and action."""

import asyncio
import contextvars
import io
import pathlib
import sys
import threading

import pytest

import flyte
import flyte.report
from flyte._context import internal_ctx
from flyte._internal.controllers import create_controller
from flyte._internal.runtime import convert, rusty
from flyte._internal.runtime import io as runtime_io
from flyte._internal.runtime._output_attribution import AttributedStream, attributed_output
from flyte.models import ActionID, RawDataPath, TaskContext


def _task_context(run_name: str, action_name: str) -> TaskContext:
    return TaskContext(
        action=ActionID(name=action_name, run_name=run_name),
        version="v1",
        raw_data_path=RawDataPath(path="/tmp/raw"),
        output_path="/tmp/out",
        run_base_dir="/tmp/base",
        report=flyte.report.Report(name=action_name),
    )


def _in_action(run_name: str, action_name: str):
    return internal_ctx().replace_task_context(_task_context(run_name, action_name))


@pytest.fixture
def stream() -> AttributedStream:
    return AttributedStream(io.StringIO())


def _out(stream: AttributedStream) -> str:
    return stream.wrapped.getvalue()


def test_lines_inside_a_task_are_prefixed(stream):
    with _in_action("r1", "a0"):
        stream.write("hello\nworld\n")
    assert _out(stream) == "[r1][a0] hello\n[r1][a0] world\n"


def test_lines_outside_a_task_pass_through(stream):
    stream.write("plain\n")
    assert _out(stream) == "plain\n"


def test_partial_writes_are_joined_before_the_prefix(stream):
    with _in_action("r1", "a0"):
        stream.write("hel")
        stream.flush()
        assert _out(stream) == ""
        stream.write("lo\nnext")
        assert _out(stream) == "[r1][a0] hello\n"
        stream.write("\n")
    assert _out(stream) == "[r1][a0] hello\n[r1][a0] next\n"


def test_blank_lines_are_kept_without_a_prefix(stream):
    with _in_action("r1", "a0"):
        stream.write("a\n\nb\n")
    assert _out(stream) == "[r1][a0] a\n\n[r1][a0] b\n"


def test_lines_already_attributed_are_not_prefixed_twice(stream):
    with _in_action("r1", "a0"):
        stream.write("[r1][a0] INFO from logging\n")
        stream.write('{"message": "x", "run_name": "r1", "action_name": "a0"}\n')
    assert _out(stream) == '[r1][a0] INFO from logging\n{"message": "x", "run_name": "r1", "action_name": "a0"}\n'


def test_end_of_invocation_writes_out_a_trailing_partial_line(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with _in_action("r1", "a0"):
        with attributed_output("r1", "a0"):
            print("no newline", end="")
            assert isinstance(sys.stdout, AttributedStream)
            assert sys.stdout.wrapped.getvalue() == ""
    assert sys.stdout.wrapped.getvalue() == "[r1][a0] no newline\n"


def test_a_very_long_partial_line_is_written_out(stream):
    with _in_action("r1", "a0"):
        stream.write("x" * 10000)
    assert _out(stream) == "[r1][a0] " + "x" * 10000 + "\n"


def test_stream_attributes_pass_through():
    class Tty(io.StringIO):
        def isatty(self):
            return True

        def fileno(self):
            return 7

        @property
        def encoding(self):
            return "utf-8"

    wrapped = AttributedStream(Tty())
    assert wrapped.isatty() is True
    assert wrapped.fileno() == 7
    assert wrapped.encoding == "utf-8"
    assert wrapped.getvalue() == ""  # delegated


def test_install_is_idempotent(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with attributed_output("r1", "a0"):
        first = sys.stdout
        with attributed_output("r1", "a1"):
            assert sys.stdout is first
    assert not isinstance(first.wrapped, AttributedStream)


@pytest.mark.asyncio
async def test_concurrent_invocations_keep_their_own_prefix(stream):
    """Two asyncio tasks interleave partial writes; each line still gets its own action."""
    step = asyncio.Event()

    async def invocation(action: str, first: bool):
        with _in_action("r1", action):
            if first:
                stream.write(f"{action} part one, ")
                step.set()
                await asyncio.sleep(0.01)
            else:
                await step.wait()
                stream.write(f"{action} whole line\n")
            stream.write(f"{action} part two\n")

    await asyncio.gather(invocation("a0", True), invocation("a1", False))

    lines = _out(stream).splitlines()
    assert sorted(lines) == sorted(
        ["[r1][a0] a0 part one, a0 part two", "[r1][a1] a1 whole line", "[r1][a1] a1 part two"]
    )


def test_threads_that_copy_the_context_get_its_prefix(stream):
    def worker():
        stream.write("from a thread\n")

    with _in_action("r1", "a0"):
        ctx = contextvars.copy_context()
    t = threading.Thread(target=ctx.run, args=(worker,))
    t.start()
    t.join()
    plain = threading.Thread(target=worker)
    plain.start()
    plain.join()

    assert _out(stream) == "[r1][a0] from a thread\nfrom a thread\n"


env = flyte.TaskEnvironment(name="output-attribution")
gate = asyncio.Event()


@env.task
async def chatty(tag: str) -> str:
    print(f"{tag} start")
    if tag == "first":
        print("first half ", end="")
        gate.set()
        await asyncio.sleep(0.05)
        print("of a line")
    else:
        await gate.wait()
    print(f"{tag} end", file=sys.stderr)
    return tag


@env.task
def sync_chatty(tag: str) -> str:
    print(f"{tag} from a sync task")
    return tag


async def _run(task, tmp_path: pathlib.Path, action: str, tag: str):
    inputs = await convert.convert_from_native_to_inputs(task.interface, tag)
    (tmp_path / action / "out").mkdir(parents=True)
    input_path = runtime_io.inputs_path(base_path=str(tmp_path / action))
    await runtime_io.upload_inputs(inputs, input_path)
    await rusty.run_task(
        task=task,
        controller=create_controller("local"),
        org="o",
        project="p",
        domain="d",
        run_name="run-1",
        name=action,
        raw_data_path=str(tmp_path / action / "raw"),
        output_path=str(tmp_path / action / "out"),
        run_base_dir=str(tmp_path / action),
        version="v1",
        input_path=input_path,
    )


@pytest.mark.asyncio
async def test_reusable_container_attributes_interleaved_prints(tmp_path: pathlib.Path, monkeypatch):
    """Two concurrent invocations in one process, as in a reusable container, interleave prints."""
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    gate.clear()

    await asyncio.gather(
        _run(chatty, tmp_path, "a0", "first"),
        _run(chatty, tmp_path, "a1", "second"),
        _run(sync_chatty, tmp_path, "a2", "third"),
    )

    stdout_lines = [line for line in out.getvalue().splitlines() if "[run-1]" in line]
    assert "[run-1][a0] first start" in stdout_lines
    assert "[run-1][a0] first half of a line" in stdout_lines
    assert "[run-1][a1] second start" in stdout_lines
    assert "[run-1][a2] third from a sync task" in stdout_lines
    assert "[run-1][a0] first end" in err.getvalue().splitlines()
    assert "[run-1][a1] second end" in err.getvalue().splitlines()
    # No print() line went out without its action.
    for text in (out.getvalue(), err.getvalue()):
        for tag in ("first start", "second start", "third from a sync task", "first end", "second end"):
            for line in text.splitlines():
                if tag in line:
                    assert line.startswith("[run-1][a"), line
