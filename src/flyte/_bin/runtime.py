"""
Flyte runtime module, this is the entrypoint script for the Flyte runtime.

Caution: Startup time for this module is very important, as it is the entrypoint for the Flyte runtime.
Refrain from importing any modules here. If you need to import any modules, do it inside the main function.
"""

import asyncio
import os
import sys
from typing import List

import click

from flyte.models import PathRewrite

ACTION_NAME = "ACTION_NAME"
RUN_NAME = "RUN_NAME"
PROJECT_NAME = "FLYTE_INTERNAL_EXECUTION_PROJECT"
DOMAIN_NAME = "FLYTE_INTERNAL_EXECUTION_DOMAIN"
ORG_NAME = "_U_ORG_NAME"
ENDPOINT_OVERRIDE = "_U_EP_OVERRIDE"
RUN_OUTPUT_BASE_DIR = "_U_RUN_BASE"
FLYTE_ENABLE_VSCODE_KEY = "_F_E_VS"

_UNION_EAGER_API_KEY_ENV_VAR = "_UNION_EAGER_API_KEY"
_F_PATH_REWRITE = "_F_PATH_REWRITE"
_F_USE_RUST_CONTROLLER = "_F_USE_RUST_CONTROLLER"


@click.group()
def _pass_through():
    pass


@_pass_through.command("a0")
@click.option("--inputs", "-i", required=True)
@click.option("--outputs-path", "-o", required=True)
@click.option("--version", "-v", required=True)
@click.option("--run-base-dir", envvar=RUN_OUTPUT_BASE_DIR, required=True)
@click.option("--raw-data-path", "-r", required=False)
@click.option("--checkpoint-path", "-c", required=False)
@click.option("--prev-checkpoint", "-p", required=False)
@click.option("--name", envvar=ACTION_NAME, required=False)
@click.option("--run-name", envvar=RUN_NAME, required=False)
@click.option("--run-start-time", required=False)
@click.option("--project", envvar=PROJECT_NAME, required=False)
@click.option("--domain", envvar=DOMAIN_NAME, required=False)
@click.option("--org", envvar=ORG_NAME, required=False)
@click.option("--debug", envvar=FLYTE_ENABLE_VSCODE_KEY, type=click.BOOL, required=False)
@click.option("--interactive-mode", type=click.BOOL, required=False)
@click.option("--image-cache", required=False)
@click.option("--tgz", required=False)
@click.option("--pkl", required=False)
@click.option("--dest", required=False)
@click.option("--resolver", required=False)
@click.argument(
    "resolver-args",
    type=click.UNPROCESSED,
    nargs=-1,
)
@click.pass_context
def main(
    ctx: click.Context,
    run_name: str,
    name: str,
    run_start_time: str,
    project: str,
    domain: str,
    org: str,
    debug: bool,
    interactive_mode: bool,
    image_cache: str,
    version: str,
    inputs: str,
    run_base_dir: str,
    outputs_path: str,
    raw_data_path: str,
    checkpoint_path: str,
    prev_checkpoint: str,
    tgz: str,
    pkl: str,
    dest: str,
    resolver: str,
    resolver_args: List[str],
):
    sys.path.insert(0, ".")

    import faulthandler
    import signal

    import flyte
    import flyte._utils as utils
    import flyte.errors
    import flyte.storage as storage
    from flyte._initialize import init_in_cluster
    from flyte._internal.controllers import create_controller
    from flyte._internal.imagebuild.image_builder import ImageCache
    from flyte._internal.runtime.entrypoints import load_and_run_task
    from flyte._logging import logger
    from flyte.models import ActionID, CheckpointPaths, CodeBundle, RawDataPath

    logger.info("Registering faulthandler for SIGUSR1")
    faulthandler.register(signal.SIGUSR1)

    logger.info(f"Initializing flyte runtime - version {flyte.__version__}")
    assert org, "Org is required for now"
    assert project, "Project is required"
    assert domain, "Domain is required"
    assert run_name, f"Run name is required {run_name}"
    assert name, f"Action name is required {name}"

    if run_name.startswith("{{"):
        run_name = os.getenv("RUN_NAME", "")
    if name.startswith("{{"):
        name = os.getenv("ACTION_NAME", "")

    from datetime import datetime, timezone

    parsed_run_start_time: datetime | None = None
    if run_start_time and not run_start_time.startswith("{{"):
        raw = run_start_time.rstrip()
        # tolerate trailing "Z" — datetime.fromisoformat only handles it on 3.11+
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed_run_start_time = datetime.fromisoformat(raw)
            if parsed_run_start_time.tzinfo is None:
                parsed_run_start_time = parsed_run_start_time.replace(tzinfo=timezone.utc)
            else:
                parsed_run_start_time = parsed_run_start_time.astimezone(timezone.utc)
        except ValueError:
            logger.warning(f"Could not parse --run-start-time {run_start_time!r}; falling back to current UTC time.")
            parsed_run_start_time = None

    logger.warning(f"Flyte runtime started for action {name} with run name {run_name}")

    if debug and name == "a0":
        from flyte._debug.vscode import _start_vscode_server

        asyncio.run(_start_vscode_server(ctx))

    bundle = None
    if tgz or pkl:
        bundle = CodeBundle(tgz=tgz, pkl=pkl, destination=dest, computed_version=version)

    controller_kwargs = init_in_cluster(org=org, project=project, domain=domain)
    # Controller is created with the same kwargs as init, so that it can be used to run tasks
    # Use Rust controller if env var is set, otherwise default to Python controller
    use_rust = os.getenv(_F_USE_RUST_CONTROLLER, "").lower() in ("1", "true", "yes")
    if use_rust:
        try:
            import flyte_controller_base  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                f"{_F_USE_RUST_CONTROLLER}=1 was set but `flyte_controller_base` is not installed. "
                "Install it with `pip install flyte[rust-controller]`. "
                "For development, run `make dev-rs-dist` from the repo root."
            ) from e
    controller_type = "rust" if use_rust else "remote"
    controller = create_controller(ct=controller_type, **controller_kwargs)  # type: ignore[arg-type]

    ic = ImageCache.from_transport(image_cache) if image_cache else None

    path_rewrite_cfg = os.getenv(_F_PATH_REWRITE, None)
    path_rewrite = None
    if path_rewrite_cfg:
        potential_path_rewrite = PathRewrite.from_str(path_rewrite_cfg)
        if storage.exists_sync(potential_path_rewrite.new_prefix):
            path_rewrite = potential_path_rewrite
            logger.info(f"Path rewrite configured for {path_rewrite.new_prefix}")
        else:
            logger.error(
                f"Path rewrite failed for path {potential_path_rewrite.new_prefix}, "
                f"not found, reverting to original path {potential_path_rewrite.old_prefix}"
            )

    # Create a coroutine to load the task and run it
    task_coroutine = load_and_run_task(
        resolver=resolver,
        resolver_args=resolver_args,
        action=ActionID(name=name, run_name=run_name, project=project, domain=domain, org=org),
        raw_data_path=RawDataPath(path=raw_data_path, path_rewrite=path_rewrite),
        checkpoint_paths=CheckpointPaths(prev_checkpoint_path=prev_checkpoint, checkpoint_path=checkpoint_path),
        code_bundle=bundle,
        input_path=inputs,
        output_path=outputs_path,
        run_base_dir=run_base_dir,
        version=version,
        controller=controller,
        image_cache=ic,
        interactive_mode=interactive_mode or debug,
        run_start_time=parsed_run_start_time,
    )
    # Create a coroutine to watch for errors
    controller_failure = controller.watch_for_errors()

    # Run both coroutines concurrently and wait for first to finish and cancel the other
    async def _run_and_stop():
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(flyte.errors.silence_polling_error)
        try:
            await utils.run_coros(controller_failure, task_coroutine)
            await controller.stop()
        except (flyte.errors.RuntimeSystemError, flyte.errors.RuntimeUserError) as e:
            from flyte._internal.runtime.convert import convert_from_native_to_error
            from flyte._internal.runtime.io import upload_error

            logger.error(f"Flyte runtime failed for action {name} with run name {run_name}, error: {e}")
            err = convert_from_native_to_error(e)
            path = await upload_error(err.err, outputs_path, recoverable=err.recoverable)
            logger.error(f"Run {run_name} Action {name} failed with error: {err}. Uploaded error to {path}")
            await controller.stop()

    asyncio.run(_run_and_stop())
    logger.warning(f"Flyte runtime completed for action {name} with run name {run_name}")
    for h in logger.handlers:
        h.flush()
    sys.stdout.flush()
    _quiet_shutdown()


def _quiet_shutdown() -> None:
    """Close fsspec filesystem sessions and silence late shutdown noise.

    gcsfs (and any other fsspec.AsyncFileSystem) registers a weakref/atexit
    finalizer that calls `asyn.sync(loop, session.close, timeout=0.1)` during
    interpreter shutdown. The 0.1s window is hard-coded, and when aiohttp's
    connector close races interpreter teardown the timeout raises and dumps a
    multi-frame traceback to stderr — purely cosmetic but trips test runners
    that assert on clean stderr (e.g. TestTaskExecutionLogs flakiness).

    Workaround: close the sessions ourselves *before* atexit gets a turn so
    the finalizer has nothing to do, and wrap stderr with a filter that drops
    the known-noise lines if something still leaks through. Both steps are
    best-effort — failures here must never mask the real task outcome.

    Tracking upstream:
      - gcsfs:   https://github.com/fsspec/gcsfs/issues/379
      - aiohttp: https://github.com/aio-libs/aiohttp/issues/1925

    TODO: remove once the storage layer routes all `gs://` traffic through
    obstore (no gcsfs import → no atexit finalizer → no shutdown noise).
    See _OBSTORE_SUPPORTED_PROTOCOLS in storage/_storage.py.
    """
    try:
        import fsspec  # already imported by the storage layer; this is cheap
        from fsspec.asyn import AsyncFileSystem
    except Exception:
        return

    # Close any registered async filesystems' sessions eagerly.
    for proto in ("gs", "s3", "abfs", "abfss"):
        try:
            fs = fsspec.filesystem(proto)
        except Exception:
            continue
        if not isinstance(fs, AsyncFileSystem):
            continue
        # gcsfs exposes `close_session(loop, session)`; other backends may not.
        # Try the documented API, then fall back to direct session.close.
        close = getattr(fs, "close_session", None)
        try:
            if close is not None and getattr(fs, "loop", None) and getattr(fs, "session", None):
                close(fs.loop, fs.session)
            elif getattr(fs, "session", None) is not None and getattr(fs.session, "closed", True) is False:
                # Last resort: drop the reference so atexit can't find it.
                fs.session = None
        except Exception:
            continue

    # Belt-and-suspenders: drop the specific shutdown traceback lines if any
    # remaining finalizer still emits them. Active only after our cleanup runs.
    _install_shutdown_stderr_filter()


def _install_shutdown_stderr_filter() -> None:
    """Wrap sys.stderr so well-known gcsfs/aiohttp shutdown noise is dropped."""
    real_stderr = sys.stderr

    class _Filter:
        _drop_markers = (
            "gcsfs/core.py",
            "fsspec/asyn.py",
            "aiohttp/connector.py",
            "asyncio.exceptions.CancelledError",
            "weakref._exitfunc",
        )

        def write(self, s: str) -> int:
            if any(m in s for m in self._drop_markers):
                return len(s)
            return real_stderr.write(s)

        def __getattr__(self, name: str):
            return getattr(real_stderr, name)

    sys.stderr = _Filter()  # type: ignore[assignment]


if __name__ == "__main__":
    _pass_through()
