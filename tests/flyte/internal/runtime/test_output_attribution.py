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
        stream.write("par")
        stream.flush()
        assert _out(stream) == ""
        stream.write("tial\nnext")
        assert _out(stream) == "[r1][a0] partial\n"
        stream.write("\n")
    assert _out(stream) == "[r1][a0] partial\n[r1][a0] next\n"


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
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with _in_action("r1", "a0"):
        with attributed_output("r1", "a0"):
            print("no newline", end="")
            assert isinstance(sys.stdout, AttributedStream)
            assert out.getvalue() == ""
    assert out.getvalue() == "[r1][a0] no newline\n"


def test_a_log_record_after_a_partial_line_is_not_joined_onto_it(stream):
    with _in_action("r1", "a0"):
        stream.write("progress: ")
        stream.write("[r1][a0] INFO a log record\n")
        stream.write('{"message": "x", "action_name": "a0"}\n')
        stream.write("50%\n")
    assert _out(stream) == (
        '[r1][a0] progress: \n[r1][a0] INFO a log record\n{"message": "x", "action_name": "a0"}\n[r1][a0] 50%\n'
    )


def test_trailing_partial_line_is_written_out_after_the_stream_was_replaced(monkeypatch):
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with _in_action("r1", "a0"):
        with attributed_output("r1", "a0"):
            print("no newline", end="")
            # Something else, a capture fixture say, puts its own stream in place mid-invocation.
            sys.stdout = io.StringIO()
    assert out.getvalue() == "[r1][a0] no newline\n"


def test_partial_line_written_after_its_invocation_ended_is_written_out(monkeypatch):
    """A thread the invocation started can write after it ended; the next invocation to end writes it out."""
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    other = attributed_output("r1", "a1")
    other.__enter__()
    with _in_action("r1", "a0"):
        with attributed_output("r1", "a0"):
            pass
        print("late", end="")
    assert out.getvalue() == ""
    other.__exit__(None, None, None)
    assert out.getvalue() == "[r1][a0] late\n"


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


def test_streams_are_restored_when_the_last_invocation_ends(monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    first = attributed_output("r1", "a0")
    second = attributed_output("r1", "a1")
    first.__enter__()
    wrapper = sys.stdout
    second.__enter__()
    # The first invocation ends while the second still runs: the wrapper stays for the second.
    first.__exit__(None, None, None)
    assert sys.stdout is wrapper
    second.__exit__(None, None, None)
    assert sys.stdout is out
    assert sys.stderr is err


def test_a_stream_replaced_during_an_invocation_is_left_in_place(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    replacement = io.StringIO()
    with attributed_output("r1", "a0"):
        sys.stdout = replacement
    assert sys.stdout is replacement


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
