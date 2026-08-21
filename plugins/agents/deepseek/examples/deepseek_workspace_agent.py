"""Give the agent a real workspace — DeepSeek Harness editing files on Flyte.

This is what the harness is built for. Its own tools (local bash, the string
editor) act on a working directory, so pointing ``workspace=`` at a real
directory turns the agent loose on actual files instead of a scratch temp dir.

The pipeline is ordinary Flyte:

    seed_repo --> fix_the_code (agent) --> the patched directory

- ``seed_repo`` produces a ``flyte.io.Dir`` — here a tiny project with a failing
  function, but just as easily a checkout, a dbt project, or a data drop.
- ``fix_the_code`` downloads it, hands the local path to ``run_agent`` as the
  workspace, and lets the agent read/edit/run whatever it needs. Its Flyte-task
  tool (``run_tests``) is published into that same workspace, so the agent can
  verify its own work — and each verification is a durable, cached child action.
- The edited directory is uploaded back as the task's output, so the change is a
  first-class, versioned artifact rather than a side effect in a dead container.

Everything the workspace-less examples get still applies: retries resume the
conversation, tool calls are durable child actions, and the timeline lands in the
task report.

Run:  flyte run deepseek_workspace_agent.py fix_pipeline
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import asyncio
import pathlib
import subprocess
import tempfile

import flyte
from flyte.io import Dir

from flyteplugins.agents.deepseek import run_agent, tool

env = flyte.TaskEnvironment(
    "deepseek-workspace",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=[flyte.Secret(key="deepseek_api_key", as_env_var="DEEPSEEK_API_KEY")],
    image=flyte.Image.from_debian_base(name="deepseek-workspace").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-deepseek"]
    ),
)

# A deliberately broken implementation: ``median`` doesn't average the two middle
# values on even-length input, so the second test fails.
_BROKEN_SOURCE = """\
# Return the median of a list of numbers.
def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
"""

_TESTS = """\
from stats import median


def test_odd_length():
    assert median([3, 1, 2]) == 2


def test_even_length():
    assert median([4, 1, 3, 2]) == 2.5
"""


@env.task
async def seed_repo() -> Dir:
    """Produce the little project the agent will be asked to fix."""
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="repo-"))
    (workdir / "stats.py").write_text(_BROKEN_SOURCE)
    (workdir / "test_stats.py").write_text(_TESTS)
    return await Dir.from_local(str(workdir))


@tool
@env.task(retries=2)
async def run_tests(directory: str) -> str:
    """Run the test suite in a directory and return the pytest output."""
    done = await asyncio.to_thread(
        subprocess.run,
        ["python", "-m", "pytest", "-q", directory],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    return f"exit code {done.returncode}\n{done.stdout}\n{done.stderr}".strip()


@env.task(report=True, retries=3)
async def fix_the_code(repo: Dir) -> Dir:
    """Agent task: let the harness read, edit and verify real files."""
    local_path = await repo.download()

    await run_agent(
        "The test suite in this directory is failing. Read the code, find the bug in "
        "stats.py, fix it, and use the run_tests tool to confirm every test passes. "
        "Reply with a one-line summary of the fix.",
        tools=[run_tests],
        instructions=(
            "You are a careful software engineer. Make the smallest correct change, and "
            "verify with run_tests before you finish."
        ),
        model="deepseek-v4-flash",
        # The harness's bash and editor act here; the run_tests shim is published
        # into this same directory, so the agent can call it like any command.
        workspace=local_path,
    )

    # The agent edited the files in place — hand the patched directory downstream.
    return await Dir.from_local(str(local_path))


@env.task(report=True, retries=3)
async def fix_pipeline() -> str:
    """Seed a broken project, let the agent fix it, then verify independently."""
    patched = await fix_the_code(await seed_repo())
    # Verify with our own task rather than trusting the agent's account of itself.
    return await run_tests(await patched.download())


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(fix_pipeline)
    print(f"View at: {run.url}")
    run.wait()
    print(f"Verification:\n{run.outputs()}")
