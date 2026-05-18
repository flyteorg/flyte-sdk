"""FlyteLauncher — Hydra Launcher plugin that dispatches jobs to Flyte.

For the ``@hydra.main`` entry point:

    python train.py hydra/launcher=flyte hydra.launcher.mode=remote

For sweeps:

    python train.py --multirun \\
        hydra/launcher=flyte hydra.launcher.mode=remote \\
        optimizer.lr=0.001,0.01,0.1 training.epochs=10,20

    python train.py --multirun \\
        hydra/launcher=flyte hydra.launcher.wait=false \\
        optimizer.lr=0.001,0.01,0.1

    python train.py --multirun \\
        hydra/launcher=flyte hydra.launcher.wait_max_workers=64 \\
        optimizer.lr=0.001,0.01,0.1

Local mode runs jobs sequentially in-process (identical to Hydra's built-in
BasicLauncher). Remote mode prints each Flyte run URL as soon as the run is
submitted, then optionally waits for all submitted runs concurrently.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from flyte import system_logger as log
from omegaconf import DictConfig, open_dict

from hydra.core.utils import JobReturn, JobStatus, configure_log, run_job, setup_globals
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction


class FlyteHydraRunResult:
    """Remote Flyte run plus resolved output for Hydra sweepers."""

    def __init__(self, run, value) -> None:
        self.run = run
        self.value = value

    @property
    def url(self) -> str:
        return self.run.url

    def __float__(self) -> float:
        return float(self.value)

    def __iter__(self):
        if isinstance(self.value, (list, tuple)):
            return iter(self.value)
        return iter((self.value,))

    def __getattr__(self, name: str):
        return getattr(self.run, name)

    def __repr__(self) -> str:
        return f"FlyteHydraRunResult(url={self.url!r}, value={self.value!r})"


class FlyteLauncher(Launcher):
    """Hydra launcher that runs each sweep job as a Flyte execution."""

    def __init__(
        self,
        mode: str = "remote",
        wait: bool = True,
        wait_max_workers: int | None = 32,
    ) -> None:
        self.mode = mode
        self.wait = wait
        self.wait_max_workers = wait_max_workers
        self.config: DictConfig | None = None
        self.task_function: TaskFunction | None = None
        self.hydra_context: HydraContext | None = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def launch(
        self,
        job_overrides: Sequence[Sequence[str]],
        initial_job_idx: int,
    ) -> Sequence[JobReturn]:
        setup_globals()
        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        Path(str(self.config.hydra.sweep.dir)).mkdir(parents=True, exist_ok=True)

        is_sweep = len(job_overrides) > 1
        if self.mode == "remote":
            return self._launch_remote(job_overrides, initial_job_idx, is_sweep)
        return self._launch_local(job_overrides, initial_job_idx, is_sweep)

    def _sweep_config(self, overrides: Sequence[str], job_idx: int) -> DictConfig:
        cfg = self.hydra_context.config_loader.load_sweep_config(self.config, list(overrides))
        with open_dict(cfg):
            cfg.hydra.job.id = str(job_idx)
            cfg.hydra.job.num = job_idx
        return cfg

    def _run_one(self, overrides: Sequence[str], job_idx: int, is_sweep: bool) -> JobReturn:
        log.info(f"\t#{job_idx} : {' '.join(overrides)}")
        return run_job(
            hydra_context=self.hydra_context,
            task_function=self.task_function,
            config=self._sweep_config(overrides, job_idx),
            job_dir_key="hydra.sweep.dir" if is_sweep else "hydra.run.dir",
            job_subdir_key="hydra.sweep.subdir" if is_sweep else None,
        )

    def _launch_local(
        self,
        job_overrides: Sequence[Sequence[str]],
        initial_job_idx: int,
        is_sweep: bool,
    ) -> list[JobReturn]:
        results = []
        for idx, overrides in enumerate(job_overrides):
            ret = self._run_one(overrides, initial_job_idx + idx, is_sweep)
            results.append(ret)
            configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        return results

    def _launch_remote(
        self,
        job_overrides: Sequence[Sequence[str]],
        initial_job_idx: int,
        is_sweep: bool,
    ) -> list[JobReturn]:
        from flyte._run import _run_mode_var

        _run_mode_var.set("remote")
        job_returns: list[JobReturn] = []
        multiple_jobs = len(job_overrides) > 1

        try:
            # Submit all jobs before blocking — platform handles parallelism.
            for idx, overrides in enumerate(job_overrides):
                job_idx = initial_job_idx + idx
                ret = self._run_one(overrides, job_idx, is_sweep)
                job_returns.append(ret)
                self._show_submitted_run(ret, job_idx, multiple_jobs)
        finally:
            _run_mode_var.set(None)

        if self.wait:
            self._wait_for_remote_runs(job_returns)
        else:
            log.info("Submitted %d Flyte runs; not waiting for completion.", len(job_returns))

        return job_returns

    def _show_submitted_run(self, ret: JobReturn, job_idx: int, multiple_jobs: bool) -> None:
        """Print a Flyte run URL immediately after submission, before waiting."""
        flyte_run = ret._return_value
        try:
            url = getattr(flyte_run, "url", None)
        except Exception:
            log.debug("Could not read Flyte run URL for job %s.", job_idx, exc_info=True)
            return

        if url is None:
            return

        prefix = f"Flyte run submitted [{job_idx}]" if multiple_jobs else "Flyte run submitted"
        print(f"{prefix}: {url}")

    def _wait_for_remote_runs(self, job_returns: list[JobReturn]) -> None:
        """Wait for all submitted Flyte runs at the same time."""
        from flyte.models import ActionPhase

        runs = [(ret, ret._return_value) for ret in job_returns if getattr(ret._return_value, "wait", None) is not None]
        if not runs:
            return

        worker_count = self._wait_worker_count(len(runs))
        log.info(
            "Waiting for %d Flyte runs to complete with %d worker threads.",
            len(runs),
            worker_count,
        )

        def _wait_one(flyte_run):
            # Each wait tracks one run; quiet=True avoids multiple Rich
            # progress renderers fighting each other in parallel.
            flyte_run.wait(quiet=True)
            outputs = flyte_run.outputs() if flyte_run.phase == ActionPhase.SUCCEEDED else None
            return flyte_run.phase, outputs

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_result = {executor.submit(_wait_one, flyte_run): (ret, flyte_run) for ret, flyte_run in runs}
            for future in as_completed(future_to_result):
                ret, flyte_run = future_to_result[future]
                try:
                    phase, outputs = future.result()
                except Exception as exc:
                    ret.return_value = exc
                    ret.status = JobStatus.FAILED
                    continue

                if phase == ActionPhase.SUCCEEDED:
                    if outputs is not None:
                        value = outputs[0] if len(outputs) == 1 else outputs
                        ret.return_value = FlyteHydraRunResult(flyte_run, value)
                    ret.status = JobStatus.COMPLETED
                    continue

                url = getattr(flyte_run, "url", flyte_run)
                ret.return_value = RuntimeError(f"Flyte run {url} finished in phase {phase}.")
                ret.status = JobStatus.FAILED

    def _wait_worker_count(self, run_count: int) -> int:
        if self.wait_max_workers is None:
            return run_count
        return min(run_count, max(1, self.wait_max_workers))
