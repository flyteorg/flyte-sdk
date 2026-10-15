import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

import flyte
import flyte.errors
from flyte._internal.apple_silicon._bridge import Bridge, BridgeServer
from flyte._internal.apple_silicon._client import host_execution_requested, run_on_host
from flyte._internal.apple_silicon._hostexec import forwardable_env, rewrite_endpoints
from flyte._internal.apple_silicon._token import mint_token
from flyte._internal.runtime.resources_serde import get_apple_silicon_execution_env

SECRET = "test-secret"

# Stand-ins for the flyte entrypoints a host environment provides.
FAKE_A0 = """#!{python}
import json, os, sys
print("argv", json.dumps(sys.argv[1:]))
keys = ("FLYTE_AWS_ENDPOINT", "_F_APPLE_SILICON_ON_HOST", "MY_VAR", "KUBERNETES_SERVICE_HOST")
print("env", json.dumps({{k: os.environ.get(k) for k in keys}}))
sys.exit(int(os.environ.get("EXIT_WITH", "0")))
"""
FAKE_FSERVE = """#!{python}
import os, time
time.sleep(float(os.environ.get("LOAD_SECONDS", "0")))  # an app loading its model before it listens
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"served on the host"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", int(os.environ["_F_APPLE_SILICON_HOST_PORT"])), H).serve_forever()
"""


class FakeEnvCache:
    def __init__(self, root: Path):
        self.root = root
        self.requests = []

    def ensure(self, python, packages, log):
        self.requests.append((python, packages))
        env_dir = self.root / "env"
        (env_dir / "bin").mkdir(parents=True, exist_ok=True)
        for name, src in (("a0", FAKE_A0), ("fserve", FAKE_FSERVE)):
            path = env_dir / "bin" / name
            path.write_text(src.format(python=sys.executable))
            path.chmod(0o755)
        return env_dir


def make_bridge(tmp_path):
    b = Bridge(SECRET, ["unused"], tmp_path / "logs", host_platform="darwin/arm64")
    b.env_cache = FakeEnvCache(tmp_path)
    server = BridgeServer(("127.0.0.1", 0), b)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    b.url = f"http://127.0.0.1:{server.server_address[1]}"
    return b, server


def container_env(bridge, workload, **extra):
    return {
        "FLYTE_APPLE_SILICON_ENDPOINT": bridge.url,
        "FLYTE_APPLE_SILICON_WORKLOAD": workload,
        "FLYTE_APPLE_SILICON_TOKEN": mint_token(SECRET, workload),
        "_F_APPLE_SILICON_EXECUTION": "host",
        "_F_APPLE_SILICON_HOST_PACKAGES": json.dumps(["torch"]),
        "FLYTE_AWS_ENDPOINT": "http://rustfs-svc.flyte:9000",
        "KUBERNETES_SERVICE_HOST": "10.43.0.1",
        "PATH": "/usr/local/bin",
        **extra,
    }


def test_task_runs_on_the_host_and_mirrors_its_exit_code(tmp_path, capfd):
    bridge, server = make_bridge(tmp_path)
    try:
        code = run_on_host("a0", ["--inputs", "s3://x"], container_env(bridge, "wl-task", MY_VAR="1", EXIT_WITH="3"))
        assert code == 3
        assert bridge.env_cache.requests == [(f"{sys.version_info.major}.{sys.version_info.minor}", ["torch"])]
        log = capfd.readouterr().err
        assert 'argv ["--inputs", "s3://x"]' in log
        env = json.loads(log.split("[apple-silicon] env ", 1)[1].splitlines()[0])
        assert env["FLYTE_AWS_ENDPOINT"] == "http://localhost:30002"  # rewritten to the devbox's host port
        assert env["_F_APPLE_SILICON_ON_HOST"] == "1"  # so the host entrypoint runs instead of delegating again
        assert env["MY_VAR"] == "1"
        assert env["KUBERNETES_SERVICE_HOST"] is None  # container-only environment stays behind
    finally:
        server.shutdown()
        bridge.shutdown()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_app_port_is_forwarded_to_the_host_process(tmp_path):
    bridge, server = make_bridge(tmp_path)
    listen = _free_port()
    result = {}
    t = threading.Thread(
        target=lambda: result.setdefault(
            "code", run_on_host("fserve", [], container_env(bridge, "app-x", PORT=str(listen)))
        ),
        daemon=True,
    )
    t.start()
    try:
        body = None
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{listen}/", timeout=2) as resp:
                    body = resp.read()
                    break
            except OSError:
                time.sleep(0.1)
        assert body == b"served on the host"
    finally:
        bridge.release("app-x")  # the app is stopped: its host process is torn down
        t.join(10)
        server.shutdown()
    assert "code" in result


def test_host_execution_needs_the_opt_in_and_a_bridge():
    assert host_execution_requested(
        {"_F_APPLE_SILICON_EXECUTION": "host", "FLYTE_APPLE_SILICON_ENDPOINT": "http://h:1"}
    )
    assert not host_execution_requested({"_F_APPLE_SILICON_EXECUTION": "host"})
    assert not host_execution_requested({"FLYTE_APPLE_SILICON_ENDPOINT": "http://h:1"})
    # On the host itself the entrypoint must run the workload, not delegate again.
    assert not host_execution_requested(
        {
            "_F_APPLE_SILICON_EXECUTION": "host",
            "FLYTE_APPLE_SILICON_ENDPOINT": "http://h:1",
            "_F_APPLE_SILICON_ON_HOST": "1",
        }
    )


def test_env_forwarding_and_rewrites():
    env = forwardable_env(
        {
            "PATH": "/x",
            "HOME": "/root",
            "MY_SECRET_NAME": "v",
            "FOO_SERVICE_PORT": "80",
            "FLYTE_APPLE_SILICON_TOKEN": "t",
        }
    )
    assert env == {"MY_SECRET_NAME": "v"}
    assert rewrite_endpoints({"_U_EP_OVERRIDE": "flyte-binary-http.flyte:8090"}) == {
        "_U_EP_OVERRIDE": "localhost:30080"
    }


def test_apple_silicon_runs_on_the_host_by_default():
    image = flyte.Image.from_debian_base().with_pip_packages("torch", "numpy")
    resources = flyte.Resources(gpu="AppleSilicon")
    assert get_apple_silicon_execution_env(resources, image, None) == {
        "_F_APPLE_SILICON_EXECUTION": "host",
        "_F_APPLE_SILICON_HOST_PACKAGES": json.dumps(["torch", "numpy"]),
    }
    # Opting out keeps the workload in its container, calling the host's MLX daemon.
    assert get_apple_silicon_execution_env(resources, image, {"FLYTE_APPLE_SILICON_EXECUTION": "container"}) == {}
    # Other accelerators, and no accelerator, carry no host-execution environment.
    assert get_apple_silicon_execution_env(flyte.Resources(gpu="A100:1"), image, None) == {}
    assert get_apple_silicon_execution_env(flyte.Resources(cpu=1), image, None) == {}
    assert get_apple_silicon_execution_env(None, image, None) == {}
    assert not hasattr(flyte, "Apple Silicon")


def test_host_app_reports_the_host_port(monkeypatch):
    from flyte.app import AppEnvironment

    app = AppEnvironment(name="a", port=8080)
    assert app.get_port().port == 8080
    monkeypatch.setenv("_F_APPLE_SILICON_HOST_PORT", "51234")
    assert app.get_port().port == 8080  # only on the host itself
    monkeypatch.setenv("_F_APPLE_SILICON_ON_HOST", "1")
    assert app.get_port().port == 51234


def test_app_port_opens_only_once_the_host_app_listens(tmp_path):
    bridge, server = make_bridge(tmp_path)
    listen = _free_port()
    env = container_env(bridge, "app-slow", PORT=str(listen), LOAD_SECONDS="2")
    t = threading.Thread(target=lambda: run_on_host("fserve", [], env), daemon=True)
    t.start()
    try:
        time.sleep(1.0)  # the host app is still "loading"
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", listen), timeout=0.5).close()
        body = None
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{listen}/", timeout=2) as resp:
                    body = resp.read()
                    break
            except OSError:
                time.sleep(0.1)
        assert body == b"served on the host"
    finally:
        bridge.release("app-slow")
        t.join(10)
        server.shutdown()
