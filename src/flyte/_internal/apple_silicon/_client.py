"""
The container side of the Apple Silicon accelerator.

When the backend schedules a workload that requested `gpu="AppleSilicon"` onto an Apple Silicon devbox, it hands the
container the bridge endpoint and a token scoped to the workload (`FLYTE_APPLE_SILICON_*`, plus `OPENAI_BASE_URL` /
`OPENAI_API_KEY` for unmodified OpenAI clients). Before any user code runs, the Flyte runtime attaches: it asks
the bridge to start the workload's daemon and waits until the model serves, so a missing model or an
unreachable bridge fails the workload up front with a named error. While the workload runs it renews the
lease and copies the daemon's log into the container's own, so it shows up next to the workload's logs.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from flyte._logging import logger

ENDPOINT_ENV = "FLYTE_APPLE_SILICON_ENDPOINT"
WORKLOAD_ENV = "FLYTE_APPLE_SILICON_WORKLOAD"
TOKEN_ENV = "FLYTE_APPLE_SILICON_TOKEN"
MODEL_ENV = "FLYTE_APPLE_SILICON_MODEL"
START_TIMEOUT_ENV = "FLYTE_APPLE_SILICON_START_TIMEOUT"

HEARTBEAT_INTERVAL = 10.0
_LOG_PREFIX = "[apple-silicon] "


def _error(code: str | None, message: str) -> Exception:
    import flyte.errors

    cls = flyte.errors.APPLE_SILICON_ERRORS_BY_CODE.get(code or "", flyte.errors.AppleSiliconDaemonStartFailedError)
    return cls(message)


class AppleSiliconAttachment:
    """
    A workload's hold on its host MLX daemon. Keep it for the workload's lifetime and `close()` it at the end,
    which releases the daemon.
    """

    def __init__(self, endpoint: str, workload_id: str, token: str, snapshot: dict[str, Any], reacquire: bool = True):
        self.endpoint = endpoint
        # Re-acquire a daemon the bridge lost (e.g. after a bridge restart). Host execution must not: a lost host
        # process is gone, and acquiring would start an unrelated daemon.
        self.reacquire = reacquire
        self.workload_id = workload_id
        self.token = token
        self.snapshot = snapshot
        self.model: str | None = snapshot.get("model")
        self._log_offset = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._keepalive, name="flyte-apple-silicon-heartbeat", daemon=True)

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 10.0):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                payload = json.loads(e.read() or b"{}")
            except ValueError:
                payload = {}
            return e.code, payload

    def _keepalive(self) -> None:
        announced_failure = False
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            try:
                self._drain_logs()
                status, snap = self._call("POST", f"/apple-silicon/v1/workloads/{self.workload_id}/heartbeat")
                if status == 404 and self.reacquire:
                    # The bridge restarted (or reaped us): take the daemon back.
                    status, snap = self._call(
                        "POST",
                        f"/apple-silicon/v1/workloads/{self.workload_id}/acquire",
                        {"model": self.model, "timeout": 5},
                    )
                    self._log_offset = 0
                if status == 503 and not announced_failure:
                    announced_failure = True
                    code = snap.get("error_code") or "DAEMON_START_FAILED"
                    logger.error(f"[{code}] {snap.get('message') or 'the MLX daemon stopped serving'}")
            except (urllib.error.URLError, OSError) as e:
                logger.warning(
                    f"[BRIDGE_UNREACHABLE] lost contact with the Apple Silicon bridge at {self.endpoint}: {e}"
                )

    def _drain_logs(self) -> None:
        status, payload = self._call(
            "GET", f"/apple-silicon/v1/workloads/{self.workload_id}/logs?offset={self._log_offset}"
        )
        if status != 200:
            return
        data = payload.get("data") or ""
        self._log_offset = payload.get("offset", self._log_offset)
        if data:
            out = "".join(f"{_LOG_PREFIX}{line}\n" for line in data.splitlines())
            sys.stderr.write(out)
            sys.stderr.flush()

    def start(self) -> "AppleSiliconAttachment":
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop the heartbeat, flush the daemon's last log lines, and tear the daemon down."""
        self._stop.set()
        try:
            self._drain_logs()
            self._call("DELETE", f"/apple-silicon/v1/workloads/{self.workload_id}")
        except (urllib.error.URLError, OSError):
            pass  # the bridge reaps the daemon once the lease lapses


def attach_from_env(
    environ: Mapping[str, str] | None = None, start_timeout: float | None = None
) -> AppleSiliconAttachment | None:
    """
    Attach to the host MLX daemon if the backend scheduled this workload as an Apple Silicon workload; otherwise do
    nothing.

    Raises:
        flyte.errors.AppleSiliconError: the bridge is unreachable, or the daemon could not start or load its model.
    """
    env = os.environ if environ is None else environ
    endpoint = env.get(ENDPOINT_ENV, "").rstrip("/")
    if not endpoint:
        return None
    workload_id = env.get(WORKLOAD_ENV, "")
    token = env.get(TOKEN_ENV, "")
    model = env.get(MODEL_ENV) or None
    timeout = start_timeout or float(env.get(START_TIMEOUT_ENV) or 900)

    if environ is None:
        os.environ.setdefault("OPENAI_BASE_URL", f"{endpoint}/v1")
        os.environ.setdefault("OPENAI_API_KEY", token)

    logger.info(f"Attaching to the Apple Silicon bridge at {endpoint} for workload {workload_id}")
    attachment = AppleSiliconAttachment(endpoint, workload_id, token, {})
    started = time.monotonic()
    try:
        status, snap = attachment._call(
            "POST",
            f"/apple-silicon/v1/workloads/{workload_id}/acquire",
            {"model": model, "timeout": timeout},
            timeout=timeout + 30,
        )
        while status == 202 and time.monotonic() - started < timeout:
            status, snap = attachment._call(
                "POST",
                f"/apple-silicon/v1/workloads/{workload_id}/acquire",
                {"model": model, "timeout": 30},
                timeout=60,
            )
    except (urllib.error.URLError, OSError) as e:
        raise _error(
            "BRIDGE_UNREACHABLE",
            f"Container could not reach the Apple Silicon endpoint at {endpoint} ({e}); "
            "restart the bridge with `flyte start devbox --apple-silicon` on the host.",
        ) from e

    attachment._drain_logs()
    if status == 401 or status == 403:
        raise _error("BRIDGE_UNREACHABLE", f"the Apple Silicon bridge rejected this workload's token: {snap}")
    if status != 200:
        error = snap.get("error") if isinstance(snap.get("error"), dict) else {}
        code = snap.get("error_code") or error.get("code")
        message = snap.get("message") or error.get("message") or f"the MLX daemon is {snap.get('state', 'unknown')}"
        raise _error(code, message)

    attachment.snapshot = snap
    attachment.model = snap.get("model")
    logger.info(
        f"Apple Silicon daemon serving for {workload_id} (model: {attachment.model or 'per request'}, "
        f"pid {snap.get('pid')}, ready in {time.monotonic() - started:.1f}s); OpenAI endpoint {endpoint}/v1"
    )
    return attachment.start()


_HOST_DONE = ("exited", "oom_killed", "start_failed", "torn_down")


def host_execution_requested(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this container should hand its entrypoint to the Mac (gpu="AppleSilicon")."""
    env = os.environ if environ is None else environ
    return (
        env.get("_F_APPLE_SILICON_EXECUTION") == "host"
        and bool(env.get(ENDPOINT_ENV))
        and not env.get("_F_APPLE_SILICON_ON_HOST")
    )


def _accepts_connections(host: str, port: int) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


async def _proxy_port(listen_port: int, host: str, port: int) -> None:
    """Forward the container's app port to the app process running on the Mac."""
    import asyncio

    async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    async def handle(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
        try:
            up_r, up_w = await asyncio.open_connection(host, port)
        except OSError:
            client_w.close()
            return
        await asyncio.gather(pipe(client_r, up_w), pipe(up_r, client_w))

    server = await asyncio.start_server(handle, "0.0.0.0", listen_port)
    async with server:
        await server.serve_forever()


def run_on_host(entrypoint: str, argv: list[str], environ: Mapping[str, str] | None = None) -> int:
    """
    Run this workload's entrypoint (`a0` or `fserve`) natively on the Mac and mirror it: stream its log, forward
    an abort, proxy an app's port, and return its exit code. The devbox must have been started with
    `flyte start devbox --apple-silicon`.

    Raises:
        flyte.errors.AppleSiliconError: the bridge is unreachable, refuses the request, or could not build the host
            environment.
    """
    import signal
    import urllib.parse

    env = os.environ if environ is None else environ
    endpoint = env[ENDPOINT_ENV].rstrip("/")
    workload_id = env.get(WORKLOAD_ENV, "")
    attachment = AppleSiliconAttachment(endpoint, workload_id, env.get(TOKEN_ENV, ""), {}, reacquire=False)
    spec = {
        "entrypoint": entrypoint,
        "argv": argv,
        "env": dict(env),
        "packages": json.loads(env.get("_F_APPLE_SILICON_HOST_PACKAGES") or "[]"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    logger.info(f"Running {entrypoint} for {workload_id} natively on the Mac (Apple Silicon host execution)")
    try:
        status, snap = attachment._call("POST", f"/apple-silicon/v1/workloads/{workload_id}/exec", spec, timeout=60)
    except (urllib.error.URLError, OSError) as e:
        raise _error(
            "BRIDGE_UNREACHABLE",
            f"Container could not reach the Apple Silicon endpoint at {endpoint} ({e}); "
            "restart the bridge with `flyte start devbox --apple-silicon` on the host.",
        ) from e
    if status != 200:
        error = snap.get("error") if isinstance(snap.get("error"), dict) else {}
        raise _error(
            error.get("code"),
            error.get("message") or f"the bridge refused to run the workload (HTTP {status})",
        )

    def on_signal(signum: int, _frame: Any) -> None:
        attachment.close()  # tears the host process down
        os._exit(128 + signum)

    if threading.current_thread() is threading.main_thread():  # entrypoints call this from the main thread
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
    attachment.start()  # lease heartbeat

    proxying = False
    while True:
        try:
            attachment._drain_logs()
            status, snap = attachment._call("GET", f"/apple-silicon/v1/workloads/{workload_id}")
        except (urllib.error.URLError, OSError) as e:
            logger.warning(f"[BRIDGE_UNREACHABLE] lost contact with the Apple Silicon bridge at {endpoint}: {e}")
            time.sleep(2)
            continue
        if status == 404:
            # Torn down from the host side (stopped app, bridge restart): nothing left to mirror.
            attachment._stop.set()
            logger.warning(f"The host process for {workload_id} is gone; the bridge no longer tracks it")
            return 1
        state = snap.get("state")
        host = urllib.parse.urlsplit(endpoint).hostname or "host.docker.internal"
        if (
            entrypoint == "fserve"
            and state == "running"
            and not proxying
            and snap.get("port")
            # Only open the container's port once the app on the host accepts connections: Knative treats an open
            # port as ready, and requests sent while the app still loads its model would fail.
            and _accepts_connections(host, int(snap["port"]))
        ):
            import asyncio

            listen = int(env.get("PORT") or 8080)
            upstream = int(snap["port"])
            threading.Thread(target=lambda: asyncio.run(_proxy_port(listen, host, upstream)), daemon=True).start()
            logger.info(f"Forwarding the app's port {listen} to the host process on {host}:{upstream}")
            proxying = True
        if state in _HOST_DONE:
            attachment._stop.set()
            attachment._drain_logs()
            if state == "start_failed":
                raise _error(snap.get("error_code"), snap.get("message") or "host execution failed to start")
            code = snap.get("exit_code")
            logger.info(f"The host process for {workload_id} finished ({state}, exit code {code})")
            return int(code) if code is not None else 1
        time.sleep(1)
