"""Example: Durable execution with mid-pipeline failure and cached recovery (Agent SDK).

Same pattern as example_durable_execution.py but using the Agent SDK path
(use_agent_sdk=True). The Agent SDK autonomously generates, tests, and fixes
code — all sandboxes created during this process are cached.

On retry after a mid-pipeline failure:
- The agent re-runs, but sandbox image builds and test executions hit cache
- The execution `Sandbox` from result.run() also hits cache
- The retry effectively resumes from the failure point

See example_durable_execution.py for detailed comments on the caching mechanism.
"""

import logging
import os
import tempfile
from pathlib import Path

import flyte
from flyte._image import PythonWheels
from flyte.errors import RuntimeUserError
from flyte.io import File
from flyte.sandbox import sandbox_environment

from flyteplugins.codegen import AutoCoderAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("flyteplugins.codegen.auto_coder_agent").setLevel(logging.INFO)

env = flyte.TaskEnvironment(
    name="durable-execution-agent-example",
    secrets=[
        flyte.Secret(key="samhita_anthropic_api_key", as_env_var="ANTHROPIC_API_KEY"),
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
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent.parent / "dist",
                package_name="flyte",
                pre=True,
            ),
            name="durable-execution-agent",
        )
    ),
    depends_on=[sandbox_environment],
)

# cache="auto" flows to sandboxes created by the Agent SDK hooks.
# When the agent runs pytest, the PreToolUse hook intercepts it and runs tests
# in a sandbox — that task gets cache="auto".
agent = AutoCoderAgent(
    name="durable-csv-agent",
    use_agent_sdk=True,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    retries=2,
    timeout=600,
    cache="auto",
)


@env.task(cache="auto")
async def prepare_data() -> File:
    """Create sample CSV data. Cached — only runs once across all attempts."""
    csv_content = """date,product,quantity,price
2024-01-01,Widget A,10,25.50
2024-01-02,Widget B,5,30.00
2024-01-03,Widget A,8,25.50
2024-01-04,Widget C,12,15.00
2024-01-05,Widget B,3,30.00"""

    csv_file = Path(tempfile.gettempdir()) / "durable_sales_data_agent.csv"
    csv_file.write_text(csv_content)
    return await File.from_local(str(csv_file))


@env.task(retries=2)
async def generate_and_run_with_retries(csv_file: File) -> dict[str, float | int]:
    """Multi-phase pipeline that fails midway and recovers via cached traces.

    Attempt 0:
      - Agent SDK autonomously generates code, writes tests, and iterates.
        Each pytest invocation is intercepted by hooks and run in a sandbox
        task with cache="auto".
      - result.run() executes the code in a cached sandbox.
      - A simulated failure is raised after both phases complete.

    Attempt 1 (retry):
      - Agent SDK re-runs, but sandboxes (image builds, test
        runs) hit cache — the agent completes much faster.
      - result.run()'s sandbox also hits cache — instant.
      - No failure — the task succeeds.
    """
    attempt = os.environ["FLYTE_ATTEMPT_NUMBER"]
    print(f"[Attempt {attempt}] Starting Agent SDK pipeline...")

    # ── Phase 1: Agent SDK code generation ─────────────────────────────────
    # The agent autonomously generates, tests, and fixes code. Each test
    # execution creates a sandbox with cache="auto".
    #
    # Attempt 0: sandboxes run fresh
    # Attempt 1: sandboxes hit cache -> agent completes faster
    print(f"[Attempt {attempt}] Phase 1: Agent SDK generating code...")
    result = await agent.generate.aio(
        prompt="Read the CSV and compute total_revenue, total_units, and row_count.",
        samples={"csv_data": csv_file},
        outputs={"total_revenue": float, "total_units": int, "row_count": int},
    )

    if not result.success:
        raise RuntimeError(f"Agent SDK code generation failed: {result.error}")

    print(f"[Attempt {attempt}] Phase 1 complete: {len(result.detected_packages)} packages detected")

    # ── Phase 2: Execute on real data ──────────────────────────────────────
    # Attempt 0: runs fresh
    # Attempt 1: sandbox hits cache -> returns instantly
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

    # Simulated failure
    if attempt == "0":
        raise RuntimeUserError(
            code="SIMULATED_FAILURE",
            message="""Simulated transient failure after sandbox execution.
            All sandboxes completed and are cached.
            On retry, they will hit cache and complete instantly.""",
        )

    return {
        "total_revenue": total_revenue,
        "total_units": total_units,
        "row_count": row_count,
    }


@env.task
async def durable_execution_agent_workflow() -> dict[str, float | int]:
    """Workflow: durable execution with Agent SDK, mid-pipeline failure, cached recovery."""
    csv_file = await prepare_data()
    return await generate_and_run_with_retries(csv_file)


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(durable_execution_agent_workflow)
    print(f"\nRun URL: {run.url}")
