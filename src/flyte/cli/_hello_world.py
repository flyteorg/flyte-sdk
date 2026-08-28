"""Built-in examples for `flyte run hello-world`, `flyte deploy hello-world-task`, and
`flyte deploy hello-world-app`.

Zero-setup first steps: the SDK carries the source of two small examples, writes them to a
scratch directory, and runs or deploys them. Nothing has to exist in the user's current
directory, and the printed path is a real file they can copy into a project of their own.

```bash
flyte run hello-world           # run the task example on the configured cluster
flyte run --local hello-world   # run it on this machine
flyte deploy hello-world-task   # deploy the same example's environment
flyte deploy hello-world-app    # deploy a minimal FastAPI app
```
"""

from __future__ import annotations

import getpass
import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, cast

import rich_click as click

from . import _common as common
from ._deploy import HELLO_WORLD_APP_CMD, HELLO_WORLD_TASK_CMD, DeployArguments, DeployEnvCommand
from ._run import HELLO_WORLD_CMD, RunArguments, RunTaskCommand

TASK_SOURCE = '''"""Your first Flyte workflow.

`main` fans `line` out over a list of inputs with `flyte.map` -- each one is its own
action -- and averages what comes back.

Run it with:

    flyte run hello_world.py main
    flyte run --local hello_world.py main

Deploy it with:

    flyte deploy hello_world.py env
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

APP_SOURCE = '''"""Your first Flyte app: a small FastAPI service.

`/` serves the landing page in index.html, which points at `/docs` -- FastAPI's
interactive API docs, where every endpoint below can be tried in the browser.

Deploy it with:

    flyte deploy hello_world_app.py app_env
"""

from pathlib import Path

import flyte
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from flyte.app.extras import FastAPIAppEnvironment

INDEX_HTML = Path(__file__).parent / "index.html"

app = FastAPI(
    title="Flyte Hello World App",
    description="A first Flyte app. Try the endpoints below.",
    version="1.0.0",
)

app_env = FastAPIAppEnvironment(
    name="hello-world-app",
    app=app,
    description="A minimal FastAPI app to try out Flyte apps.",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    # index.html is not a Python module, so the bundler only ships it because it is
    # named here. Paths are relative to this file.
    include=("index.html",),
)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Landing page. Links to the interactive API docs at /docs."""
    return FileResponse(INDEX_HTML)


@app.get("/hello")
async def hello(name: str = "world") -> dict[str, str]:
    """Say hello. Try `name=flyte`."""
    return {"message": f"Hello, {name}!"}


@app.get("/line/{x}")
async def line(x: int) -> dict[str, int]:
    """The same y = 2x + 5 that the hello-world task computes."""
    return {"x": x, "y": 2 * x + 5}


@app.post("/mean")
async def mean(numbers: list[float]) -> dict[str, float]:
    """Average a list of numbers. Try `[1, 2, 3]`."""
    if not numbers:
        raise HTTPException(status_code=400, detail="numbers must not be empty")
    return {"mean": sum(numbers) / len(numbers)}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    return {"status": "healthy"}
'''

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Flyte Hello World App</title>
    <style>
      :root { color-scheme: light dark; }
      body {
        margin: 0 auto;
        padding: 3rem 1.5rem;
        max-width: 42rem;
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
        line-height: 1.6;
      }
      h1 { margin-bottom: 0.25rem; }
      p.lede { margin-top: 0; opacity: 0.75; }
      a.docs {
        display: inline-block;
        margin: 1rem 0 2rem;
        padding: 0.6rem 1.1rem;
        border-radius: 0.5rem;
        background: #4b48d6;
        color: #fff;
        font-weight: 600;
        text-decoration: none;
      }
      ul { padding-left: 1.1rem; }
      li { margin-bottom: 0.4rem; }
      code {
        padding: 0.1rem 0.35rem;
        border-radius: 0.25rem;
        background: rgba(127, 127, 127, 0.18);
        font-size: 0.9em;
      }
    </style>
  </head>
  <body>
    <h1>Flyte Hello World App</h1>
    <p class="lede">A minimal FastAPI app, deployed with <code>flyte deploy hello-world-app</code>.</p>

    <a class="docs" href="/docs">Open the interactive API docs &rarr;</a>

    <p>Every endpoint can be tried from <a href="/docs">/docs</a>, or directly:</p>
    <ul>
      <li><a href="/hello?name=flyte"><code>GET /hello?name=flyte</code></a> &mdash; say hello</li>
      <li><a href="/line/4"><code>GET /line/4</code></a> &mdash; y = 2x + 5, the line the hello-world task computes</li>
      <li><code>POST /mean</code> with a body of <code>[1, 2, 3]</code> &mdash; average a list of numbers</li>
      <li><a href="/health"><code>GET /health</code></a> &mdash; health check</li>
    </ul>

    <p>The source of this app is on your machine &mdash; the path was printed when you deployed it.
    Copy it into a project of your own and start editing.</p>
  </body>
</html>
"""


def _scratch_root() -> Path:
    """Per-user parent for the scratch directories.

    Per-user so a shared `/tmp` on Linux cannot hand one user a directory another user owns.
    """
    try:
        user = getpass.getuser()
    except Exception:
        user = "default"
    return Path(tempfile.gettempdir()) / f"flyte-hello-world-{user}"


@dataclass(frozen=True)
class Example:
    """A built-in example the CLI can materialize and then run or deploy.

    Each example gets its own directory: the code bundle is rooted there, so anything
    sharing the directory would be packaged along with it.
    """

    slug: str
    filename: str
    module: str
    source: str
    #: Non-Python files written alongside the source, by name.
    extra_files: Mapping[str, str] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        """Scratch directory this example is written to.

        Stable across invocations so the path printed to the user stays valid.
        """
        return _scratch_root() / self.slug

    def materialize(self, directory: Path | None = None) -> Path:
        """Write the example out and return the path of its Python file.

        Rewritten on every invocation: the files belong to the installed SDK, so a copy
        left by a previous version should never win.
        """
        directory = directory or self.directory
        directory.mkdir(parents=True, exist_ok=True)
        for name, contents in self.extra_files.items():
            (directory / name).write_text(contents)
        path = directory / self.filename
        path.write_text(self.source)
        return path

    def load(self, path: Path, name: str) -> Any:
        """Import the materialized file and return one object from it.

        The object has to come from this file rather than from an equivalent copy inside
        the `flyte` package: code bundling resolves a task or an app through its module's
        location, and a module in site-packages is not something the remote side can load.
        """
        spec = importlib.util.spec_from_file_location(self.module, path)
        if spec is None or spec.loader is None:
            raise click.ClickException(f"Could not load the {self.slug} example from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module] = module
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as e:
            # The example is only as installable as its imports. Say which package is
            # missing rather than surfacing a traceback from a file the user never wrote.
            raise click.ClickException(
                f"The {self.slug} example needs `{e.name}` installed locally: pip install {e.name}"
            ) from e
        obj = getattr(module, name, None)
        if obj is None:
            raise click.ClickException(f"The {self.slug} example has no object named {name!r}")
        return obj


TASK_EXAMPLE = Example(
    slug="task",
    filename="hello_world.py",
    module="hello_world",
    source=TASK_SOURCE,
)

APP_EXAMPLE = Example(
    slug="app",
    filename="hello_world_app.py",
    module="hello_world_app",
    source=APP_SOURCE,
    extra_files={"index.html": INDEX_HTML},
)


def endpoint_configured(ctx: click.Context) -> bool:
    """Whether a run or a deploy has anywhere to go."""
    if os.getenv("FLYTE_API_KEY"):
        return True
    obj = ctx.obj
    if getattr(obj, "endpoint", None):
        return True
    platform = getattr(getattr(obj, "config", None), "platform", None)
    return bool(getattr(platform, "endpoint", None))


NO_ENDPOINT = (
    "No Flyte endpoint is configured.\nRun `flyte create config --endpoint <your-endpoint>` to connect to a cluster."
)


def _output_format(ctx: click.Context) -> common.OutputFormat:
    return cast(common.OutputFormat, getattr(ctx.obj, "output_format", "table"))


def _announce_source(ctx: click.Context, source_path: Path) -> None:
    common.get_console().print(
        common.get_panel(
            "Hello World",
            f"Using the built-in example from {source_path}\nCopy it into your own project to start editing.",
            _output_format(ctx),
        )
    )


def _ensure_cli_config(ctx: click.Context) -> None:
    """Give the context a CLIConfig when `run`/`deploy` was invoked without going through main."""
    if ctx.obj is None:
        import flyte.config

        ctx.obj = common.CLIConfig(config=flyte.config.auto(), ctx=ctx)


class HelloWorldCommand(RunTaskCommand):
    """`flyte run hello-world`: a `RunTaskCommand` that says where its source came from."""

    def __init__(self, source_path: Path, *args, **kwargs):
        self.source_path = source_path
        super().__init__(*args, **kwargs)

    def invoke(self, ctx: click.Context):
        if not self.run_args.local and not endpoint_configured(ctx):
            # Someone running their first workflow may not have a cluster yet. Running locally
            # is the useful thing to do, but say so rather than doing it silently.
            self.run_args.local = True
            common.get_console().print(
                common.get_panel(
                    "Hello World",
                    f"{NO_ENDPOINT}\nRunning locally in the meantime.",
                    _output_format(ctx),
                )
            )

        _announce_source(ctx, self.source_path)
        return super().invoke(ctx)


class HelloWorldDeployCommand(DeployEnvCommand):
    """`flyte deploy hello-world-*`: a `DeployEnvCommand` that says where its source came
    from, and what to do with what it just deployed."""

    def __init__(self, source_path: Path, next_step: str, *args, **kwargs):
        self.source_path = source_path
        self.next_step = next_step
        super().__init__(*args, **kwargs)

    def invoke(self, ctx: click.Context):
        # Unlike a run, a deploy has no local mode -- without an endpoint there is nothing to
        # do but say which command fixes it.
        if not self.deploy_args.dry_run and not endpoint_configured(ctx):
            raise click.UsageError(NO_ENDPOINT)

        _announce_source(ctx, self.source_path)
        result = super().invoke(ctx)
        common.get_console().print(common.get_panel("Next", self.next_step, _output_format(ctx)))
        return result


def get_hello_world_command(ctx: click.Context, run_args: RunArguments) -> click.Command:
    """Build the `flyte run hello-world` command."""
    path = TASK_EXAMPLE.materialize()
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
        obj=TASK_EXAMPLE.load(path, "main"),
        run_args=run_args,
        help=f"""
Run a built-in first workflow -- no files needed.

The example fans a small computation out over a list of inputs with `flyte.map` and averages
the results. Its source is written to `{TASK_EXAMPLE.filename}` in a scratch directory and the
path is printed, so you can copy it into a project of your own.

```bash
flyte run {HELLO_WORLD_CMD}
flyte run --local {HELLO_WORLD_CMD}
```
""",
    )


def _get_deploy_command(
    ctx: click.Context,
    deploy_args: DeployArguments,
    example: Example,
    env_name: str,
    cmd_name: str,
    next_step: str,
    help: str,
) -> click.Command:
    """Build one of the `flyte deploy hello-world-*` commands."""
    _ensure_cli_config(ctx)
    path = example.materialize()
    # Same as the run command: root the bundle where the example was written.
    deploy_args.root_dir = str(path.parent)

    common.initialize_config(
        ctx,
        deploy_args.project,
        deploy_args.domain,
        deploy_args.root_dir,
        tuple(deploy_args.image) or None,
        not deploy_args.no_sync_local_sys_paths,
    )

    env = example.load(path, env_name)
    return HelloWorldDeployCommand(
        source_path=path,
        next_step=next_step,
        name=cmd_name,
        env_name=env.name,
        env=env,
        deploy_args=deploy_args,
        help=help,
    )


def get_hello_world_task_deploy_command(ctx: click.Context, deploy_args: DeployArguments) -> click.Command:
    """Build the `flyte deploy hello-world-task` command."""
    return _get_deploy_command(
        ctx,
        deploy_args,
        example=TASK_EXAMPLE,
        env_name="env",
        cmd_name=HELLO_WORLD_TASK_CMD,
        next_step="Run the deployed task with:\n  flyte run deployed-task hello_world.main",
        help=f"""
Deploy the built-in first workflow -- no files needed.

Deploys the environment behind `flyte run {HELLO_WORLD_CMD}`, so its tasks can be run from the
UI or with `flyte run deployed-task`. The source is written to `{TASK_EXAMPLE.filename}` in a
scratch directory and the path is printed, so you can copy it into a project of your own.

```bash
flyte deploy {HELLO_WORLD_TASK_CMD}
```
""",
    )


def get_hello_world_app_deploy_command(ctx: click.Context, deploy_args: DeployArguments) -> click.Command:
    """Build the `flyte deploy hello-world-app` command."""
    return _get_deploy_command(
        ctx,
        deploy_args,
        example=APP_EXAMPLE,
        env_name="app_env",
        cmd_name=HELLO_WORLD_APP_CMD,
        next_step=(
            "Open the app URL above. Its landing page links to `/docs`,\n"
            "where every endpoint can be tried in the browser."
        ),
        help=f"""
Deploy a built-in first app -- no files needed.

A minimal FastAPI service whose landing page links to `/docs`, FastAPI's interactive API docs,
where a handful of toy endpoints can be tried in the browser. The source is written to
`{APP_EXAMPLE.filename}` (with its `index.html`) in a scratch directory and the path is printed,
so you can copy it into a project of your own.

Needs `fastapi` installed locally, because the app object is built here before it is deployed.

```bash
flyte deploy {HELLO_WORLD_APP_CMD}
```
""",
    )
