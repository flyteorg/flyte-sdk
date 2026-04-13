from __future__ import annotations

from types import SimpleNamespace

from hydra.core.utils import JobReturn, JobStatus

from flyteplugins.hydra._launcher import FlyteHydraRunResult, FlyteLauncher


def test_show_submitted_run_prints_single_url(capsys) -> None:
    launcher = FlyteLauncher()
    ret = SimpleNamespace(_return_value=SimpleNamespace(url="https://flyte.example/runs/one"))

    launcher._show_submitted_run(ret, job_idx=0, multiple_jobs=False)

    assert capsys.readouterr().out == "Flyte run submitted: https://flyte.example/runs/one\n"


def test_show_submitted_run_prints_indexed_sweep_url(capsys) -> None:
    launcher = FlyteLauncher()
    ret = SimpleNamespace(_return_value=SimpleNamespace(url="https://flyte.example/runs/two"))

    launcher._show_submitted_run(ret, job_idx=3, multiple_jobs=True)

    assert capsys.readouterr().out == "Flyte run submitted [3]: https://flyte.example/runs/two\n"


def test_wait_for_remote_runs_wraps_successful_output() -> None:
    from flyte.models import ActionPhase

    class FakeRun:
        url = "https://flyte.example/runs/done"
        phase = ActionPhase.SUCCEEDED

        def wait(self, quiet: bool) -> None:
            assert quiet is True

        def outputs(self) -> list[float]:
            return [0.75]

    ret = JobReturn()
    ret.return_value = FakeRun()

    FlyteLauncher()._wait_for_remote_runs([ret])

    assert ret.status is JobStatus.COMPLETED
    assert isinstance(ret.return_value, FlyteHydraRunResult)
    assert ret.return_value.url == "https://flyte.example/runs/done"
    assert ret.return_value.value == 0.75
    assert float(ret.return_value) == 0.75


def test_wait_worker_count_caps_large_sweeps() -> None:
    launcher = FlyteLauncher(wait_max_workers=32)

    assert launcher._wait_worker_count(2) == 2
    assert launcher._wait_worker_count(1000) == 32


def test_wait_worker_count_can_be_uncapped() -> None:
    launcher = FlyteLauncher(wait_max_workers=None)

    assert launcher._wait_worker_count(1000) == 1000
