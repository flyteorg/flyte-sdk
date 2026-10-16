"""
The host-side Apple Silicon bridge: one process per devbox, run by `flyte start devbox` on an Apple Silicon Mac.

It listens on 127.0.0.1 only (containers reach it through the Docker host gateway) and serves:

    GET    /healthz                              liveness, no auth (also reports the bridge's pid and flyte version)
    GET    /apple-silicon/v1/workloads                      every workload's daemon status (bridge secret)
    POST   /apple-silicon/v1/workloads/{id}/acquire         start the workload's daemon and wait until it serves
    POST   /apple-silicon/v1/workloads/{id}/heartbeat       renew the workload's lease
    POST   /apple-silicon/v1/workloads/{id}/exec            run the workload's entrypoint on the Mac (`_hostexec`)
    GET    /apple-silicon/v1/workloads/{id}                 the workload's status (host runs: their exit code)
    GET    /apple-silicon/v1/workloads/{id}/logs?offset=N   the daemon's log from byte N
    DELETE /apple-silicon/v1/workloads/{id}                 tear the daemon down
    *      /v1/...                               OpenAI-compatible API, proxied to the workload's daemon

Workload routes and the proxy take the workload's token (`_token`) as a bearer credential, so a container
only ever reaches the daemon started for it. Each request also names the pod it comes from (the `X-Flyte-Apple-Silicon-
Instance` header): pods of one app revision share the workload's host process or daemon, which is torn down when the
last of them lets go (a DELETE, or no request for a lease period). A task's run belongs to the one pod that started
it. A workload that only sends OpenAI requests (an image without the Flyte runtime) gets its daemon lazily on first
request and loses it after it has been idle for a while.
"""

from __future__ import annotations

import argparse
import hmac
import http.client
import json
import os
import re
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ._daemon import DAEMON_START_FAILED, DEAD_STATES, Daemon, DaemonState
from ._hostexec import HostProcess, safe_name
from ._token import verify_token

BRIDGE_UNREACHABLE = "APPLE_SILICON_BRIDGE_UNREACHABLE"
HOST_EXEC_FAILED = "APPLE_SILICON_HOST_EXEC_FAILED"
UNAUTHORIZED = "UNAUTHORIZED"
FORBIDDEN = "FORBIDDEN"
INSTANCE_HEADER = "X-Flyte-Apple-Silicon-Instance"
# Large enough for any entrypoint invocation and its environment; anything bigger is a bad request.
_MAX_BODY_BYTES = 8 * 1024 * 1024

# Headers that describe one hop of a connection rather than the message, so they are not forwarded.
_HOP_HEADERS = frozenset(
    {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade"}
)
_WORKLOAD_ROUTE = re.compile(
    r"^/apple-silicon/v1/workloads/(?P<id>[^/]+)(?:/(?P<action>acquire|heartbeat|logs|exec))?$"
)
_MAX_FINISHED_RECORDS = 50


class BridgeError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass
class Workload:
    daemon: Daemon
    attached: bool = False
    last_seen: float = field(default_factory=time.monotonic)
    inflight: int = 0
    # The pods holding this workload, and when each was last heard from.
    instances: dict[str, float] = field(default_factory=dict)

    @property
    def host(self) -> bool:
        return isinstance(self.daemon, HostProcess)

    def seen(self, instance: str | None) -> None:
        now = time.monotonic()
        self.last_seen = now
        if instance is not None:
            self.instances[instance] = now


class Bridge:
    """
    Owns every workload's daemon: starts it on demand, notices when it dies, and tears it down.

    Args:
        secret: The devbox's bridge secret; workload tokens are minted from it.
        command: How to run `mlx_lm.server` on this host, or None if it can't (then only host execution works).
        log_dir: Where daemon logs go, one file per workload.
        lease_ttl: Seconds without a heartbeat before an attached workload's daemon is torn down.
        idle_ttl: Seconds without a request before an unattached workload's daemon is torn down.
        start_timeout: Seconds a daemon gets to load its model.
    """

    def __init__(
        self,
        secret: str,
        command: list[str] | None,
        log_dir: Path,
        host_platform: str,
        lease_ttl: float = 60.0,
        idle_ttl: float = 900.0,
        start_timeout: float = 900.0,
    ):
        self.secret = secret
        self.command = command
        self.log_dir = log_dir
        self.host_platform = host_platform
        self.lease_ttl = lease_ttl
        self.idle_ttl = idle_ttl
        self.start_timeout = start_timeout
        self._workloads: dict[str, Workload] = {}
        self._finished: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        from ._hostexec import _EnvCache

        self.env_cache = _EnvCache(log_dir.parent / "envs")

    def _new_daemon(self, workload_id: str, model: str | None) -> Daemon:
        daemon = Daemon(
            workload_id,
            self.command,
            self.log_dir / f"{safe_name(workload_id)}.log",
            model=model,
            start_timeout=self.start_timeout,
        )
        daemon.start()
        return daemon

    def ensure(
        self, workload_id: str, model: str | None = None, *, attach: bool = False, instance: str | None = None
    ) -> Daemon:
        """
        Return the workload's daemon, starting one if it has none. `attach` (an explicit acquire) also replaces
        a daemon that died, e.g. for a retried task; a plain OpenAI request never restarts a failed daemon, so
        the failure is reported instead of looping.
        """
        retired = None
        with self._lock:
            w = self._workloads.get(workload_id)
            if w is not None and w.host:
                raise BridgeError(409, "CONFLICT", f"workload {workload_id} runs on the host; it has no MLX daemon")
            if w is not None and w.daemon.refresh() in DEAD_STATES and attach:
                retired = self._workloads.pop(workload_id)
                w = None
            if w is None:
                w = Workload(daemon=self._new_daemon(workload_id, model))
                self._workloads[workload_id] = w
            w.attached = w.attached or attach
            w.seen(instance if attach else None)
        if retired is not None:
            self._retire(retired)
        return w.daemon

    def start_exec(self, workload_id: str, spec: dict[str, Any], instance: str = "") -> Any:
        """
        Run a workload's entrypoint on the host, replacing any finished run of the same workload. A pod that
        reconnects gets its own run back, and another replica of the same app shares it; a second pod of a task
        is refused rather than handed the first one's run.
        """
        from ._hostexec import forwardable_env, rewrite_endpoints, validate_pip_args

        entrypoint, argv, env, packages, pip_args, python, declared_port = _parse_exec_spec(spec)
        try:
            pip_args = validate_pip_args(pip_args)
        except ValueError as e:
            raise BridgeError(400, "BAD_REQUEST", str(e))
        retired = None
        with self._lock:
            existing = self._workloads.get(workload_id)
            if existing is not None and existing.host and existing.daemon.refresh() in ("preparing", "running"):
                if entrypoint == "fserve" or instance in existing.instances:
                    existing.seen(instance)
                    return existing.daemon
                raise BridgeError(
                    409,
                    "CONFLICT",
                    f"workload {workload_id} is already running on the host for another pod; an Apple Silicon task "
                    "runs as a single pod",
                )
            proc = HostProcess(
                workload_id,
                entrypoint=entrypoint,
                argv=argv,
                env=rewrite_endpoints(forwardable_env(env)),
                python=python,
                packages=packages,
                state_dir=self.log_dir.parent,
                env_cache=self.env_cache,
                app_port=entrypoint == "fserve",
                declared_port=declared_port,
                pip_args=pip_args,
            )
            retired = self._workloads.pop(workload_id, None)
            w = Workload(daemon=proc, attached=True)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            w.seen(instance)
            self._workloads[workload_id] = w
        if retired is not None:
            self._retire(retired)
        sys.stderr.write(
            f"[flyte-apple-silicon-bridge] host execution for {workload_id}: "
            f"{proc.entrypoint} {' '.join(proc.argv)[:300]}\n"
        )
        proc.start()
        return proc

    def get(self, workload_id: str) -> Workload | None:
        with self._lock:
            return self._workloads.get(workload_id)

    def touch(self, workload_id: str, instance: str | None = None) -> Workload | None:
        with self._lock:
            w = self._workloads.get(workload_id)
            if w is not None:
                w.seen(instance)
            return w

    def add_inflight(self, w: Workload, delta: int) -> None:
        # The record the request started on, even if the workload has been replaced since.
        with self._lock:
            w.inflight += delta
            w.last_seen = time.monotonic()

    def release(self, workload_id: str, instance: str | None = None) -> None:
        """Let go of the workload for `instance` (or for everyone), tearing it down once no pod holds it."""
        with self._lock:
            w = self._workloads.get(workload_id)
            if w is None:
                return
            if instance is not None:
                w.instances.pop(instance, None)
                if w.instances:
                    return
            del self._workloads[workload_id]
        self._retire(w)

    def _retire(self, w: Workload) -> None:
        # Called without the lock held: stopping a daemon can take its full grace period, and other
        # workloads' requests must not wait on it.
        w.daemon.stop()
        with self._lock:
            self._finished.append(w.daemon.snapshot())
            del self._finished[:-_MAX_FINISHED_RECORDS]

    def reap(self) -> None:
        """Tear down daemons whose workload has gone away."""
        now = time.monotonic()
        expired = []
        with self._lock:
            for workload_id, w in list(self._workloads.items()):
                w.daemon.refresh()
                if w.inflight > 0:
                    continue
                if w.attached:
                    # Pods that stopped calling (killed without a DELETE) let go; the last one takes the workload.
                    w.instances = {i: t for i, t in w.instances.items() if now - t <= self.lease_ttl}
                    if not w.instances and now - w.last_seen > self.lease_ttl:
                        expired.append(self._workloads.pop(workload_id))
                elif now - w.last_seen > self.idle_ttl:
                    expired.append(self._workloads.pop(workload_id))
        for w in expired:
            self._retire(w)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            live = []
            for workload_id, w in self._workloads.items():
                w.daemon.refresh()
                live.append({**w.daemon.snapshot(), "attached": w.attached})
            return live + list(reversed(self._finished))

    def shutdown(self) -> None:
        """Tear every workload down at once, so the bridge exits well within `flyte delete devbox`'s patience."""
        with self._lock:
            workloads = list(self._workloads.values())
            self._workloads.clear()
        threads = [threading.Thread(target=self._retire, args=(w,), daemon=True) for w in workloads]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15.0)


def _parse_exec_spec(
    spec: dict[str, Any],
) -> tuple[str, list[str], dict[str, str], list[str], list[str], str, int | None]:
    entrypoint = spec.get("entrypoint")
    if entrypoint not in ("a0", "fserve"):
        raise BridgeError(400, "BAD_REQUEST", "entrypoint must be a0 or fserve")
    argv, env = spec.get("argv", []), spec.get("env", {})
    packages, pip_args = spec.get("packages", []), spec.get("pip_args", [])
    if not (isinstance(argv, list) and all(isinstance(a, str) for a in argv)):
        raise BridgeError(400, "BAD_REQUEST", "argv must be a list of strings")
    if not (isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())):
        raise BridgeError(400, "BAD_REQUEST", "env must map strings to strings")
    for name, value in (("packages", packages), ("pip_args", pip_args)):
        if not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            raise BridgeError(400, "BAD_REQUEST", f"{name} must be a list of strings")
    if any("\x00" in v for v in [*argv, *env.values(), *packages, *pip_args]):
        raise BridgeError(400, "BAD_REQUEST", "arguments and environment values can't contain NUL bytes")
    python = str(spec.get("python") or "3.12")
    if not re.fullmatch(r"3\.\d{1,2}", python):
        raise BridgeError(400, "BAD_REQUEST", f"unsupported python version {python!r}")
    port = spec.get("app_port")
    declared_port = port if isinstance(port, int) and 0 < port < 65536 else None
    return entrypoint, argv, env, packages, pip_args, python, declared_port


def _error_body(code: str, message: str) -> bytes:
    # The shape OpenAI clients parse, so the code and message reach the user's exception.
    return json.dumps({"error": {"code": code, "message": message, "type": "flyte_apple_silicon_error"}}).encode()


class BridgeHandler(BaseHTTPRequestHandler):
    server: "BridgeServer"
    # HTTP/1.0: every response ends by closing the connection, which lets a streamed (SSE) completion be
    # relayed as it arrives without re-chunking it.
    protocol_version = "HTTP/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[flyte-apple-silicon-bridge] {self.address_string()} {format % args}\n")

    @property
    def bridge(self) -> Bridge:
        return self.server.bridge

    def _bearer(self) -> str:
        auth = self.headers.get("Authorization", "")
        return auth[7:].strip() if auth.lower().startswith("bearer ") else ""

    def _instance(self) -> str:
        return self.headers.get(INSTANCE_HEADER, "")[:253]

    def _authorized_workload(self) -> str:
        workload_id = verify_token(self.bridge.secret, self._bearer())
        if workload_id is None:
            raise BridgeError(401, UNAUTHORIZED, "missing or invalid Apple Silicon workload token")
        return workload_id

    def _require_admin(self) -> None:
        if not self.bridge.secret or not hmac.compare_digest(self._bearer(), self.bridge.secret):
            raise BridgeError(401, UNAUTHORIZED, "the bridge secret is required")

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, err: BridgeError) -> None:
        body = _error_body(err.code, err.message)
        self.send_response(err.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _content_length(self) -> int:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise BridgeError(400, "BAD_REQUEST", "Content-Length must be an integer")
        if length < 0 or length > _MAX_BODY_BYTES:
            raise BridgeError(413, "TOO_LARGE", f"request bodies are limited to {_MAX_BODY_BYTES} bytes")
        return length

    def _read_json(self) -> dict[str, Any]:
        length = self._content_length()
        if not length:
            return {}
        try:
            data = json.loads(self.rfile.read(length))
        except ValueError:
            raise BridgeError(400, "BAD_REQUEST", "request body must be JSON")
        if not isinstance(data, dict):
            raise BridgeError(400, "BAD_REQUEST", "request body must be a JSON object")
        return data

    def _dispatch(self) -> None:
        try:
            path = urlsplit(self.path).path
            if path == "/healthz" and self.command == "GET":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "host_platform": self.bridge.host_platform,
                        "pid": os.getpid(),
                        "flyte_version": _flyte_version(),
                    },
                )
            elif path == "/apple-silicon/v1/workloads" and self.command == "GET":
                self._require_admin()
                self._send_json(200, {"workloads": self.bridge.snapshot()})
            elif m := _WORKLOAD_ROUTE.match(path):
                self._workload_route(m.group("id"), m.group("action"))
            elif path.startswith("/v1/"):
                self._proxy()
            else:
                raise BridgeError(404, "NOT_FOUND", f"no route for {self.command} {path}")
        except BridgeError as e:
            self._send_error(e)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client went away mid-response
        except Exception as e:  # a bug must still answer, not drop the connection
            sys.stderr.write(f"[flyte-apple-silicon-bridge] error handling {self.command} {self.path}:\n")
            traceback.print_exc()
            try:
                self._send_error(BridgeError(500, "INTERNAL", f"the Apple Silicon bridge failed: {e}"))
            except OSError:
                pass

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _dispatch

    def _workload_route(self, workload_id: str, action: str | None) -> None:
        if self._authorized_workload() != workload_id:
            raise BridgeError(403, FORBIDDEN, "this token belongs to a different workload")
        instance = self._instance()

        if action == "acquire" and self.command == "POST":
            body = self._read_json()
            model = body.get("model") or None
            if model is not None and not isinstance(model, str):
                raise BridgeError(400, "BAD_REQUEST", "model must be a string")
            daemon = self.bridge.ensure(workload_id, model, attach=True, instance=instance)
            try:
                requested = float(body.get("timeout") or self.bridge.start_timeout)
            except (TypeError, ValueError):
                raise BridgeError(400, "BAD_REQUEST", "timeout must be a number")
            daemon.wait_settled(max(0.0, min(requested, self.bridge.start_timeout)))
            self._send_daemon_state(daemon)
        elif action == "heartbeat" and self.command == "POST":
            w = self.bridge.touch(workload_id, instance)
            if w is None:
                raise BridgeError(404, "NOT_FOUND", f"workload {workload_id} has no daemon; acquire it again")
            self._send_daemon_state(w.daemon)
        elif action == "logs" and self.command == "GET":
            w = self.bridge.get(workload_id)
            if w is None:
                raise BridgeError(404, "NOT_FOUND", f"workload {workload_id} has no daemon")
            try:
                offset = max(0, int(parse_qs(urlsplit(self.path).query).get("offset", ["0"])[0]))
            except ValueError:
                raise BridgeError(400, "BAD_REQUEST", "offset must be an integer")
            data, next_offset = w.daemon.read_log(offset)
            self._send_json(200, {"data": data.decode(errors="replace"), "offset": next_offset})
        elif action == "exec" and self.command == "POST":
            proc = self.bridge.start_exec(workload_id, self._read_json(), instance)
            self._send_json(200, proc.snapshot())
        elif action is None and self.command == "GET":
            w = self.bridge.touch(workload_id, instance)
            if w is None:
                raise BridgeError(404, "NOT_FOUND", f"workload {workload_id} has no daemon or host process")
            self._send_json(200, w.daemon.snapshot())
        elif action is None and self.command == "DELETE":
            self.bridge.release(workload_id, instance)
            self._send_json(200, {"workload": workload_id, "state": DaemonState.TORN_DOWN.value})
        else:
            raise BridgeError(405, "METHOD_NOT_ALLOWED", f"{self.command} is not allowed here")

    def _send_daemon_state(self, daemon: Daemon) -> None:
        snapshot = daemon.snapshot()
        if snapshot.get("execution") == "host":
            self._send_json(200, snapshot)  # host processes report their own state; the caller reads it
        elif daemon.state == DaemonState.STARTING:
            self._send_json(202, snapshot)
        elif daemon.state == DaemonState.SERVING:
            self._send_json(200, snapshot)
        else:
            self._send_json(503, snapshot)

    def _proxy(self) -> None:
        workload_id = self._authorized_workload()
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            raise BridgeError(411, "LENGTH_REQUIRED", "chunked request bodies are not supported")
        w = self.bridge.get(workload_id)
        if w is not None and w.host:
            raise BridgeError(409, "CONFLICT", f"workload {workload_id} runs on the host; it has no MLX daemon to call")
        daemon = w.daemon if w is not None else self.bridge.ensure(workload_id)
        daemon.wait_settled(self.bridge.start_timeout)
        if daemon.refresh() != DaemonState.SERVING:
            err = daemon.error
            raise BridgeError(
                503,
                err.code if err else DAEMON_START_FAILED,
                err.message if err else f"the MLX daemon for {workload_id} is {daemon.state.value}",
            )

        length = self._content_length()
        body = self.rfile.read(length) if length else None
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in _HOP_HEADERS and k.lower() not in ("authorization", "host")
        }
        w = self.bridge.get(workload_id)
        if w is None:
            raise BridgeError(503, DAEMON_START_FAILED, f"the MLX daemon for {workload_id} went away")
        self.bridge.add_inflight(w, 1)
        # Generous, but finite: a daemon that hangs mid-response must not pin the workload forever.
        conn = http.client.HTTPConnection("127.0.0.1", daemon.port, timeout=600)
        try:
            try:
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
            except OSError as e:
                daemon.refresh()
                err = daemon.error
                raise BridgeError(
                    503, err.code if err else DAEMON_START_FAILED, err.message if err else f"MLX daemon: {e}"
                )
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                # http.client has already de-chunked the body; the connection close delimits it instead.
                if k.lower() not in _HOP_HEADERS and k.lower() != "transfer-encoding":
                    self.send_header(k, v)
            self.end_headers()
            while chunk := resp.read1(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            conn.close()
            self.bridge.add_inflight(w, -1)


def _flyte_version() -> str:
    try:
        from flyte._version import __version__

        return __version__
    except ImportError:
        return "unknown"


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], bridge: Bridge):
        super().__init__(address, BridgeHandler)
        self.bridge = bridge


def serve(bridge: Bridge, host: str, port: int, reap_interval: float = 5.0) -> None:
    """Serve until SIGTERM/SIGINT, then tear down every daemon."""
    server = BridgeServer((host, port), bridge)
    stop = threading.Event()

    def reaper() -> None:
        while not stop.wait(reap_interval):
            bridge.reap()

    def on_signal(signum: int, _frame: Any) -> None:
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    threading.Thread(target=reaper, name="apple-silicon-reaper", daemon=True).start()
    sys.stderr.write(
        f"[flyte-apple-silicon-bridge] serving on http://{host}:{port} "
        f"(daemon: {' '.join(bridge.command) if bridge.command else 'unavailable'})\n"
    )
    try:
        server.serve_forever()
    finally:
        stop.set()
        bridge.shutdown()
        server.server_close()
        sys.stderr.write("[flyte-apple-silicon-bridge] stopped; all MLX daemons torn down\n")


def main(argv: list[str] | None = None) -> int:
    from . import _host

    parser = argparse.ArgumentParser(prog="python -m flyte._internal.apple_silicon._bridge", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=_host.DEFAULT_BRIDGE_PORT)
    parser.add_argument("--state-dir", type=Path, default=_host.STATE_DIR)
    parser.add_argument("--lease-ttl", type=float, default=60.0)
    parser.add_argument("--idle-ttl", type=float, default=900.0)
    parser.add_argument("--start-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    reason = _host.apple_silicon_unavailable_reason()
    if reason:
        sys.stderr.write(f"[flyte-apple-silicon-bridge] APPLE_SILICON_NOT_AVAILABLE: {reason}\n")
        return 2
    command = _host.resolve_daemon_command()
    if command is None:
        # Host execution (the default) doesn't need it; only container-mode workloads do, and they get the reason.
        sys.stderr.write(f"[flyte-apple-silicon-bridge] container mode unavailable: {_host.DAEMON_PACKAGE_MISSING}\n")

    bridge = Bridge(
        secret=_host.load_or_create_secret(args.state_dir),
        command=command,
        log_dir=args.state_dir / "logs",
        host_platform=_host.host_platform(),
        lease_ttl=args.lease_ttl,
        idle_ttl=args.idle_ttl,
        start_timeout=args.start_timeout,
    )
    serve(bridge, args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
