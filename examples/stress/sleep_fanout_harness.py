"""
Submit N copies of the `sleep_fanout` task through the `flyte run` CLI.

Each run spawns n_children core-sleep leaves in leaseworker (no task pods).
Submissions are launched with a bounded semaphore to cap in-flight TCP
connections and server-side admission pressure; we do not wait for any run
to finish — the harness returns as soon as all submissions are accepted.
"""

import argparse
import asyncio
import os
import pathlib
import shutil
import sys
import time
from datetime import timedelta

import re

RUN_URL_RE = re.compile(r"URL:\s+(\S+/runs/[^/?\s]+)")
RUN_NAME_RE = re.compile(r"Created Run:\s+([^\s]+)")
RUNS_FILE = os.getenv("FLYTE_HARNESS_RUNS_FILE")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCAL_SDK_SRC = REPO_ROOT / "src"
FLYTE_BIN = os.getenv("FLYTE_HARNESS_FLYTE_BIN") or shutil.which("flyte") or "flyte"
FORCE_LOCAL_SDK = os.getenv("FLYTE_HARNESS_FORCE_LOCAL_SDK", "").lower() in {"1", "true", "yes", "on"}


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if FORCE_LOCAL_SDK:
        existing = env.get("PYTHONPATH", "")
        local_src = str(LOCAL_SDK_SRC)
        env["PYTHONPATH"] = f"{local_src}:{existing}" if existing else local_src
    return env


async def submit_one(sem: asyncio.Semaphore, idx: int, n_children: int, sleep_duration: timedelta) -> str | None:
    async with sem:
        os.environ.setdefault("_U_USE_ACTIONS", "1")
        config = os.getenv("FLYTE_HARNESS_CONFIG", os.path.expanduser("~/.flyte/config-dogfood.yaml"))
        image_builder = os.getenv("FLYTE_HARNESS_IMAGE_BUILDER", "remote")
        project = os.getenv("FLYTE_HARNESS_PROJECT", "")
        domain = os.getenv("FLYTE_HARNESS_DOMAIN", "")
        run_env_keys = tuple(
            k
            for k in (
                "_F_MAX_QPS",
                "_F_CTRL_WORKERS",
                "_F_P_CNC",
                "_U_USE_ACTIONS",
                "_F_TRACE_SUBMIT",
                "_F_TRACE_SUBMIT_LIMIT",
            )
            if os.getenv(k)
        )

        cmd = [FLYTE_BIN, "-c", config, "--image-builder", image_builder, "run"]
        if project:
            cmd.extend(["-p", project])
        if domain:
            cmd.extend(["-d", domain])
        for key in run_env_keys:
            cmd.extend(["--env", f"{key}={os.environ[key]}"])
        cmd.extend(
            [
                "examples/stress/sleep_fanout.py",
                "sleep_fanout",
                "--n_children",
                str(n_children),
                "--sleep_duration",
                f"PT{int(sleep_duration.total_seconds())}S",
            ]
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=_subprocess_env(),
            )

            output_lines: list[str] = []
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                output_lines.append(text)

            rc = await proc.wait()
            output = "\n".join(output_lines)
            if rc != 0:
                print(f"[{idx}] submit failed rc={rc} output={output!r}", flush=True)
                return None

            url_match = RUN_URL_RE.search(output)
            if url_match:
                return url_match.group(1)

            name_match = RUN_NAME_RE.search(output)
            if name_match:
                return name_match.group(1)

            if "/runs/" in output:
                print(f"[{idx}] submit failed: partial run URL parse failure output={output!r}", flush=True)
            else:
                print(f"[{idx}] submit failed: could not parse run id from output={output!r}", flush=True)
                return None
            return None
        except Exception as e:
            cause = getattr(e, "__cause__", None)
            print(f"[{idx}] submit failed: {type(e).__name__}: {e!r} cause={cause!r}", flush=True)
            return None


async def submit_many(total: int, concurrency: int, n_children: int, sleep_duration: timedelta) -> int:
    sem = asyncio.Semaphore(concurrency)
    start = time.monotonic()
    submitted = 0
    failed = 0
    runs_file_lock = asyncio.Lock()

    async def wrapped(i: int):
        nonlocal submitted, failed
        name = await submit_one(sem, i, n_children, sleep_duration)
        if name is None:
            failed += 1
        else:
            submitted += 1
            if RUNS_FILE:
                async with runs_file_lock:
                    with open(RUNS_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{name}\n")
            print(f"submitted_run idx={i} url={name}", flush=True)
        done = submitted + failed
        if done % 100 == 0:
            elapsed = time.monotonic() - start
            rps = done / elapsed if elapsed > 0 else 0
            print(f"submitted={submitted} failed={failed} elapsed={elapsed:.1f}s rps={rps:.1f}", flush=True)

    await asyncio.gather(*(wrapped(i) for i in range(total)))

    elapsed = time.monotonic() - start
    rps = submitted / elapsed if elapsed > 0 else 0
    print(f"\nDone. submitted={submitted} failed={failed} elapsed={elapsed:.2f}s rps={rps:.2f}")
    return 1 if failed else 0


# python stress/sleep_fanout_harness.py --total 25000 --concurrency 500 --n_children 10 --sleep_seconds 10
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=500)
    parser.add_argument("--n_children", type=int, default=10)
    parser.add_argument("--sleep_seconds", type=int, default=10)
    args = parser.parse_args()

    rc = asyncio.run(
        submit_many(
            total=args.total,
            concurrency=args.concurrency,
            n_children=args.n_children,
            sleep_duration=timedelta(seconds=args.sleep_seconds),
        )
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
