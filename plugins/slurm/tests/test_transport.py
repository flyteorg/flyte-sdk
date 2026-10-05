"""Tests that exercise SSHTransport itself.

Everything in `test_slurm.py` swaps the transport out, so `submit`, `status`, `cancel`
and `tail` -- the code most likely to break against a real cluster -- had no coverage.
These drive the real class with a fake `conn.run`, so command construction, the
squeue -> sacct -> scontrol fallback chain, and the failure paths are all covered.
"""

import asyncio

import pytest

from flyteplugins.slurm.transport import SSHTransport, parse_scontrol


class _FakeResult:
    def __init__(self, stdout="", stderr="", exit_status=0):
        self.stdout, self.stderr, self.exit_status = stdout, stderr, exit_status


class _FakeConn:
    """Answers commands from a table, recording what was asked."""

    def __init__(self, responses):
        self.responses = responses
        self.commands: list[str] = []

    async def run(self, command, check=False):
        self.commands.append(command)
        for match, result in self.responses.items():
            if match in command:
                return result
        return _FakeResult()


def _transport(conn) -> SSHTransport:
    t = SSHTransport(host="login", username="flyte", private_key="KEY")

    async def _connection():
        return conn

    t._connection = _connection  # type: ignore[method-assign]
    return t


class TestStatusFallbacks:
    def test_squeue_answers_without_touching_sacct(self):
        conn = _FakeConn({"squeue": _FakeResult(stdout="900|RUNNING|None\n")})
        states = asyncio.run(_transport(conn).status(["900"]))
        assert states["900"].base_state == "RUNNING"
        assert not any("sacct" in c for c in conn.commands)

    def test_falls_back_to_sacct_for_a_finished_job(self):
        conn = _FakeConn(
            {
                "squeue": _FakeResult(stdout="", exit_status=1, stderr="slurm_load_jobs error: Invalid job id"),
                "sacct": _FakeResult(stdout="900|COMPLETED|0:0|None\n"),
            }
        )
        states = asyncio.run(_transport(conn).status(["900"]))
        assert states["900"].base_state == "COMPLETED"

    def test_falls_back_to_scontrol_when_accounting_is_off(self):
        """Without accounting a finished job leaves squeue and is unknown to sacct."""
        conn = _FakeConn(
            {
                "squeue": _FakeResult(stdout="", exit_status=1, stderr="Invalid job id specified"),
                "sacct": _FakeResult(stdout=""),
                "scontrol": _FakeResult(stdout="JobId=900 JobName=t JobState=COMPLETED ExitCode=0:0 Reason=None"),
            }
        )
        states = asyncio.run(_transport(conn).status(["900"]))
        assert states["900"].base_state == "COMPLETED"
        assert any("scontrol show job" in c for c in conn.commands)

    def test_unreachable_controller_is_not_a_missing_job(self):
        """Otherwise a running job is reported as vanished and the task fails."""
        conn = _FakeConn(
            {"squeue": _FakeResult(exit_status=1, stderr="slurm_load_jobs error: Unable to contact slurm controller")}
        )
        with pytest.raises(RuntimeError, match="could not reach Slurm"):
            asyncio.run(_transport(conn).status(["900"]))

    def test_batches_ids_into_one_squeue(self):
        conn = _FakeConn({"squeue": _FakeResult(stdout="1|RUNNING|None\n2|RUNNING|None\n")})
        asyncio.run(_transport(conn).status(["2", "1", "2"]))
        assert len([c for c in conn.commands if "squeue" in c]) == 1
        assert "-j 1,2" in conn.commands[0]


class TestCommands:
    def test_cancel_tolerates_an_already_finished_job(self):
        conn = _FakeConn({"scancel": _FakeResult(exit_status=1, stderr="scancel: error: Invalid job id 900")})
        asyncio.run(_transport(conn).cancel("900"))  # must not raise

    def test_cancel_surfaces_a_real_failure(self):
        conn = _FakeConn({"scancel": _FakeResult(exit_status=1, stderr="Access/permission denied")})
        with pytest.raises(RuntimeError, match="scancel"):
            asyncio.run(_transport(conn).cancel("900"))

    def test_tail_of_a_missing_file_is_empty(self):
        conn = _FakeConn({"tail": _FakeResult(exit_status=1, stderr="No such file")})
        assert asyncio.run(_transport(conn).tail("/nope", 10)) == ""

    def test_command_timeout_names_the_host(self):
        class _Hang(_FakeConn):
            async def run(self, command, check=False):
                await asyncio.sleep(5)

        t = _transport(_Hang({}))
        t._command_timeout = 0.05
        with pytest.raises(RuntimeError, match="timed out"):
            asyncio.run(t.status(["900"]))


def test_parse_scontrol_ignores_output_it_cannot_read():
    assert parse_scontrol("not a scontrol line") is None
    state = parse_scontrol("JobId=7 JobState=FAILED ExitCode=1:0 Reason=NonZeroExitCode")
    assert (state.job_id, state.base_state, state.exit_code) == ("7", "FAILED", "1:0")
