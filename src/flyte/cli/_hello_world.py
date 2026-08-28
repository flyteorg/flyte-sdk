"""CLI command for `flyte run hello-world`.

A zero-setup first workflow: the SDK ships the source of a small example, writes it to a
scratch directory, and runs it. Nothing has to exist in the user's current directory, and
the printed path is a real file they can copy into a project of their own.

Usage:

```bash
flyte run hello-world           # run it on the configured Flyte cluster
flyte run --local hello-world   # run it on this machine
```
"""

from __future__ import annotations

import getpass
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import rich_click as click

from . import _common as common
from ._run import HELLO_WORLD_CMD, RunArguments, RunTaskCommand

#: Name of the file written to the scratch directory, and therefore the module name the
#: task is loaded under -- the same pairing `flyte run hello_world.py main` would produce.
HELLO_WORLD_FILENAME = "hello_world.py"
HELLO_WORLD_MODULE = "hello_world"
HELLO_WORLD_TASK = "main"

HELLO_WORLD_SOURCE = '''"""Your first Flyte workflow.

`main` fans `line` out over a list of inputs with `flyte.map` -- each one is its own
action -- and averages what comes back.

Run it with:

    flyte run hello_world.py main
    flyte run --local hello_world.py main
"""

import flyte

env = flyte.TaskEnvironment(name="hello_world")


@env.task
def line(x: int) -> int:
    """Compute a single point on the line y = 2x + 5."""
    slope, intercept = 2, 5
    return slope * x + intercept


@env.task
def main(x_list: list[int] = list(range(10))) -> float:
    """Map `line` over every x, then average the results."""
    y_list = list(flyte.map(line, x_list))
    return sum(y_list) / len(y_list)


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(main).url)
'''


def hello_world_dir() -> Path:
    """Scratch directory the example is written to.

    Stable across invocations so the path we print stays valid, and per-user so a shared
    `/tmp` on Linux cannot hand one user a directory another user owns.
    """
    try:
        user = getpass.getuser()
    except Exception:
        user = "default"
    return Path(tempfile.gettempdir()) / f"flyte-hello-world-{user}"


def materialize(directory: Path | None = None) -> Path:
    """Write the example source out and return its path.

    Rewritten on every invocation: the file belongs to the installed SDK, so an older copy
    left by a previous version should never win.
    """
    directory = directory or hello_world_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / HELLO_WORLD_FILENAME
    path.write_text(HELLO_WORLD_SOURCE)
    return path


def load_task(path: Path, task_name: str = HELLO_WORLD_TASK) -> Any:
    """Import the materialized file and return the task to run.

    The task has to come from this file rather than from an equivalent copy inside the
    `flyte` package: code bundling resolves a task by its module's location, and a task
    whose module lives in site-packages is not something the remote side can load.
    """
    spec = importlib.util.spec_from_file_location(HELLO_WORLD_MODULE, path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Could not load the hello-world example from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[HELLO_WORLD_MODULE] = module
    spec.loader.exec_module(module)
    task = getattr(module, task_name, None)
    if task is None:
        raise click.ClickException(f"The hello-world example has no task named {task_name!r}")
    return task


def endpoint_configured(ctx: click.Context) -> bool:
    """Whether a remote run has anywhere to go."""
    if os.getenv("FLYTE_API_KEY"):
        return True
    obj = ctx.obj
    if getattr(obj, "endpoint", None):
        return True
    platform = getattr(getattr(obj, "config", None), "platform", None)
    return bool(getattr(platform, "endpoint", None))


class HelloWorldCommand(RunTaskCommand):
    """`flyte run hello-world`: a `RunTaskCommand` that announces where its source came from."""

    def __init__(self, source_path: Path, *args, **kwargs):
        self.source_path = source_path
        super().__init__(*args, **kwargs)

    def invoke(self, ctx: click.Context):
        console = common.get_console()
        output_format = cast(common.OutputFormat, getattr(ctx.obj, "output_format", "table"))

        if not self.run_args.local and not endpoint_configured(ctx):
            # Someone running their first workflow may not have a cluster yet. Running locally
            # is the useful thing to do, but say so rather than doing it silently.
            self.run_args.local = True
            console.print(
                common.get_panel(
                    "Hello World",
                    "No Flyte endpoint is configured, so this will run locally.\n"
                    "Run `flyte create config --endpoint <your-endpoint>` to run on a cluster.",
                    output_format,
                )
            )

        console.print(
            common.get_panel(
                "Hello World",
                f"Running the built-in example from {self.source_path}\n"
                f"Copy it into your own project to start editing.",
                output_format,
            )
        )
        return super().invoke(ctx)


def get_hello_world_command(ctx: click.Context, run_args: RunArguments) -> click.Command:
    """Build the `flyte run hello-world` command."""
    path = materialize()
    # The example lives outside the user's project, so the code bundle has to be rooted where
    # it was written -- otherwise there is nothing under the root dir to package.
    run_args.root_dir = str(path.parent)

    common.initialize_config(
        ctx,
        run_args.project,
        run_args.domain,
        run_args.root_dir,
        tuple(run_args.image) or None,
        not run_args.no_sync_local_sys_paths,
    )

    return HelloWorldCommand(
        source_path=path,
        obj_name=HELLO_WORLD_CMD,
        obj=load_task(path),
        run_args=run_args,
        help=f"""
Run a built-in first workflow -- no files needed.

The example fans a small computation out over a list of inputs with `flyte.map` and averages
the results. Its source is written to `{HELLO_WORLD_FILENAME}` in a scratch directory and the
path is printed, so you can copy it into a project of your own.

```bash
flyte run {HELLO_WORLD_CMD}
flyte run --local {HELLO_WORLD_CMD}
```
""",
    )
