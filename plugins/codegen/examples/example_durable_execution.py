"""Example: Durable execution with mid-pipeline failure and cached recovery (LLM approach).

Demonstrates how Flyte's caching and replay logs make retries resume from the failure point
rather than restarting from scratch:

1. agent.generate() runs the full LLM pipeline — internally it builds sandbox
   images and executes tests in sandboxes. With cache="auto", these are
   cached by Flyte.

2. result.run() executes the generated code in another sandbox,
   also with cache="auto".

3. A simulated failure occurs AFTER both phases complete (e.g. a downstream
   service error, OOM, or network blip during result serialization).

4. Flyte retries the task. On retry:
   - agent.generate() re-executes, but the internal sandboxes
     (image build, test execution) hit cache and the traces replay
     — the agent completes in seconds instead of minutes.
   - result.run() re-executes, but its sandbox also hits cache —
     returning the previous result instantly.
   - The pipeline effectively "resumes" past the failure point **without
     re-doing any expensive LLM calls or sandbox work**.

Key config:
- cache="auto" on AutoCoderAgent: flows to all internal sandboxes
- cache="auto" on result.run(): execution sandbox is also cached
- retries=2 on the outer Flyte task: Flyte retries on failure
- sandbox_retries=2 on the agent: sandboxes themselves retry on infra errors
"""

import logging
import os
import tempfile
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyte.io import File
from flyte.sandbox import sandbox_environment

from flyteplugins.codegen import AutoCoderAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("flyteplugins.codegen.auto_coder_agent").setLevel(logging.INFO)

env = flyte.TaskEnvironment(
    name="durable-execution-example",
    secrets=[
        flyte.Secret(key="niels_openai_api_key", as_env_var="OPENAI_API_KEY"),
    ],
    resources=flyte.Resources(cpu=2, memory="5Gi"),
    image=(
        flyte.Image.from_debian_base()
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-codegen",
                pre=True,
            ),
        )
        .with_apt_packages("git")
        .with_pip_packages(
            "git+https://github.com/flyteorg/flyte-sdk.git@86f88fece16d956e28667d3f0d8d49108c8cdd68"
        )
    ),
    depends_on=[sandbox_environment],
)

# cache="auto" flows to every sandbox created during generate().
# sandbox_retries=2 means each sandbox retries on transient infra errors.
agent = AutoCoderAgent(
    name="durable-csv-processor",
    model="gpt-4.1",
    max_iterations=5,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    sandbox_retries=2,
    timeout=600,
    cache="auto",
)


@env.task(cache="auto")
def prepare_data() -> File:
    """Create sample CSV data. Cached — only runs once across all attempts."""
    csv_content = """date,product,quantity,price
2024-01-01,Widget A,10,25.50
2024-01-02,Widget B,5,30.00
2024-01-03,Widget A,8,25.50
2024-01-04,Widget C,12,15.00
2024-01-05,Widget B,3,30.00"""

    csv_file = Path(tempfile.gettempdir()) / "durable_sales_data.csv"
    csv_file.write_text(csv_content)
    return File.from_local_sync(str(csv_file))


@env.task(retries=2)
async def generate_and_run_with_retries(csv_file: File) -> dict[str, float | int]:
    """Multi-phase pipeline that fails midway and recovers via cached traces.

    Attempt 0 (first run):
      - agent.generate() runs the full LLM loop: plan → code → image build →
        test execution. The image build and test runs are sandboxes
        with cache="auto" — Flyte caches their results.
      - result.run() executes the generated code in a sandbox,
        also cached.
      - A simulated failure is raised AFTER both phases complete.

    Attempt 1 (retry):
      - agent.generate() re-executes, but the internal sandboxes (image
        build, test runs) hit cache — the generate phase completes in seconds.
      - result.run() re-executes, but its sandbox also hits cache —
        returns the previous output instantly.
      - No failure this time — the task succeeds.
      - Total retry cost: just the LLM calls (fast), all sandbox work is skipped.
    """
    attempt = os.environ["FLYTE_ATTEMPT_NUMBER"]
    print(f"[Attempt {attempt}] Starting pipeline...")

    # ── Phase 1: Code generation ───────────────────────────────────────────
    # Internally creates sandboxes for image building and test
    # execution. With cache="auto", these are cached by Flyte.
    #
    # Attempt 0: runs fresh (builds image, executes tests)
    # Attempt 1: sandboxes hit cache → generate completes faster
    print(f"[Attempt {attempt}] Phase 1: Generating code...")
    result = await agent.generate.aio(
        prompt="Read the CSV and compute total_revenue, total_units, and row_count.",
        samples={"csv_data": csv_file},
        outputs={"total_revenue": float, "total_units": int, "row_count": int},
    )

    if not result.success:
        raise RuntimeError(f"Code generation failed: {result.error}")

    print(f"[Attempt {attempt}] Phase 1 complete: code generated, tests passed")

    # ── Phase 2: Execute on real data ──────────────────────────────────────
    # Creates a sandbox with cache="auto".
    #
    # Attempt 0: runs the generated code in a fresh sandbox
    # Attempt 1: sandbox hits cache → returns previous result instantly
    print(f"[Attempt {attempt}] Phase 2: Executing generated code...")
    total_revenue, total_units, row_count = await result.run.aio(
        name="execute-on-data",
        retries=2,
        cache="auto",
    )

    print(f"[Attempt {attempt}] Phase 2 complete: execution finished")
    print(f"  Total revenue: ${total_revenue:.2f}")
    print(f"  Total units: {total_units}")
    print(f"  Row count: {row_count}")

    # ── Simulated failure ──────────────────────────────────────────────────
    # On attempt 0: all sandbox work (phases 1 and 2) completed and is cached,
    # but a transient error occurs (e.g. network blip, OOM, downstream service).
    #
    # On attempt 1: phases 1 and 2 "replay" via cache hits (instant), then
    # this code path succeeds.
    if attempt == "0":
        raise RuntimeError(
            "Simulated transient failure after sandbox execution. "
            "All sandboxes completed and are cached. "
            "On retry, they will hit cache and complete instantly."
        )

    return {
        "total_revenue": total_revenue,
        "total_units": total_units,
        "row_count": row_count,
    }


@env.task
async def durable_execution_workflow() -> dict[str, float | int]:
    """Workflow: durable execution with mid-pipeline failure and cached recovery.

    Run once: prepare_data (cached) → generate_and_run (fails attempt 0, retries,
    succeeds attempt 1 with cached sandbox traces).

    Re-run: prepare_data hits cache, generate_and_run hits cache → instant.
    """
    csv_file = prepare_data()
    return await generate_and_run_with_retries(csv_file)


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(durable_execution_workflow)
    print(f"\nRun URL: {run.url}")
