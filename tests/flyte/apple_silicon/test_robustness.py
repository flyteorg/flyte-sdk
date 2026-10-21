"""Regression tests for the Apple Silicon bridge, client, CLI and serialization edge cases."""

import json
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import flyte
import flyte.errors
from flyte._internal.apple_silicon import _host
from flyte._internal.apple_silicon._bridge import INSTANCE_HEADER, Bridge, BridgeServer
from flyte._internal.apple_silicon._client import run_on_host
from flyte._internal.apple_silicon._daemon import read_log_lines
from flyte._internal.apple_silicon._hostexec import safe_name, validate_pip_args
from flyte._internal.apple_silicon._token import mint_token

SECRET = "robustness-secret-" + "0" * 40

FAKE_A0 = """#!{python}
import os, signal, sys, time
for i in range(int(os.environ.get("PRINT_LINES", "0"))):
    print(f"line {{i}} " + "x" * 60)
sys.stdout.flush()
if os.environ.get("KILL_SELF"):
    os.kill(os.getpid(), getattr(signal, os.environ["KILL_SELF"]))
time.sleep(float(os.environ.get("RUN_SECONDS", "0")))
sys.exit(int(os.environ.get("EXIT_WITH", "0")))
"""
FAKE_FSERVE = """#!{python}
import time
time.sleep(600)
"""


class FakeEnvCache:
    def __init__(self, root: Path, entrypoints: bool = True):
        self.root = root
        self.entrypoints = entrypoints

    def ensure(self, python, packages, log, pip_args=None):
        env_dir = self.root / "env"
        (env_dir / "bin").mkdir(parents=True, exist_ok=True)
        if self.entrypoints:
            for name, src in (("a0", FAKE_A0), ("fserve", FAKE_FSERVE)):
                path = env_dir / "bin" / name
                path.write_text(src.format(python=sys.executable))
                path.chmod(0o755)
        return env_dir


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    from flyte._internal.apple_silicon import _hostexec

    monkeypatch.setattr(_hostexec, "_port_is_free", lambda port: False)
    b = Bridge(SECRET, None, tmp_path / "logs", host_platform="darwin/arm64", lease_ttl=1.0)
    b.env_cache = FakeEnvCache(tmp_path)
    server = BridgeServer(("127.0.0.1", 0), b)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    b.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield b
    server.shutdown()
    b.shutdown()


def env_for(bridge, workload, **extra):
    return {
        "FLYTE_APPLE_SILICON_ENDPOINT": bridge.url,
        "FLYTE_APPLE_SILICON_WORKLOAD": workload,
        "FLYTE_APPLE_SILICON_TOKEN": mint_token(SECRET, workload),
        "_F_APPLE_SILICON_EXECUTION": "host",
        "_F_APPLE_SILICON_HOST_PACKAGES": "[]",
        **extra,
    }


def call(bridge, method, path, workload, body=None, instance="pod-a", raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(
        f"{bridge.url}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {mint_token(SECRET, workload)}",
            INSTANCE_HEADER: instance,
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_state(bridge, workload, states, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        w = bridge.get(workload)
        if w is not None and w.daemon.refresh() in states:
            return w
        time.sleep(0.05)
    raise AssertionError(f"{workload} never reached {states}")


# Apps: one host process per revision, shared by its replicas, torn down when the last pod lets go.


def test_app_replicas_share_one_host_process_until_the_last_lets_go(bridge):
    spec = {"entrypoint": "fserve", "argv": [], "env": {}, "packages": []}
    status, _ = call(bridge, "POST", "/apple-silicon/v1/workloads/app-x/exec", "app-x", spec, instance="pod-a")
    assert status == 200
    w = wait_state(bridge, "app-x", ("running",))
    status, second = call(bridge, "POST", "/apple-silicon/v1/workloads/app-x/exec", "app-x", spec, instance="pod-b")
    assert status == 200 and second["pid"] == w.daemon.pid  # a replica shares the run
    call(bridge, "DELETE", "/apple-silicon/v1/workloads/app-x", "app-x", instance="pod-a")
    assert bridge.get("app-x") is not None  # pod-b still holds it
    call(bridge, "DELETE", "/apple-silicon/v1/workloads/app-x", "app-x", instance="pod-b")
    assert bridge.get("app-x") is None


def test_a_second_pod_cannot_take_over_a_tasks_run(bridge):
    spec = {"entrypoint": "a0", "argv": [], "env": {"RUN_SECONDS": "30"}, "packages": []}
    assert call(bridge, "POST", "/apple-silicon/v1/workloads/t/exec", "t", spec, instance="pod-a")[0] == 200
    wait_state(bridge, "t", ("running",))
    status, body = call(bridge, "POST", "/apple-silicon/v1/workloads/t/exec", "t", spec, instance="pod-b")
    assert status == 409 and body["error"]["code"] == "CONFLICT"
    # The pod that started it reconnects to its own run.
    assert call(bridge, "POST", "/apple-silicon/v1/workloads/t/exec", "t", spec, instance="pod-a")[0] == 200


def test_pods_that_stop_calling_let_go(bridge):
    spec = {"entrypoint": "fserve", "argv": [], "env": {}, "packages": []}
    call(bridge, "POST", "/apple-silicon/v1/workloads/app-y/exec", "app-y", spec, instance="pod-a")
    w = wait_state(bridge, "app-y", ("running",))
    time.sleep(1.2)  # past the lease
    bridge.reap()
    assert bridge.get("app-y") is None
    assert w.daemon.state == "torn_down"


# Host processes: failures are reported, never hangs; nothing outlives the run.


def test_an_entrypoint_that_cannot_start_fails_instead_of_hanging(tmp_path, bridge):
    bridge.env_cache = FakeEnvCache(tmp_path / "empty", entrypoints=False)
    with pytest.raises(flyte.errors.AppleSiliconHostExecFailedError, match="could not start a0"):
        run_on_host("a0", [], env_for(bridge, "no-entrypoint"))


def test_out_of_memory_kill_is_a_named_error(bridge):
    with pytest.raises(flyte.errors.AppleSiliconDaemonOOMError):
        run_on_host("a0", [], env_for(bridge, "oom", KILL_SELF="SIGKILL"))


def test_a_signal_exit_is_reported_like_a_shell_would(bridge):
    assert run_on_host("a0", [], env_for(bridge, "term", KILL_SELF="SIGTERM")) == 128 + signal.SIGTERM


def test_the_whole_log_reaches_the_container(bridge, capfd):
    # Far more than one 64 KB log read, printed just before exiting.
    assert run_on_host("a0", [], env_for(bridge, "chatty", PRINT_LINES="5000")) == 0
    err = capfd.readouterr().err
    lines = [line for line in err.splitlines() if line.startswith("[apple-silicon] line ")]
    assert len(lines) == 5000
    assert lines[-1].startswith("[apple-silicon] line 4999 ")


def test_a_new_run_starts_a_new_log(bridge, capfd):
    run_on_host("a0", [], env_for(bridge, "rerun", PRINT_LINES="3"))
    capfd.readouterr()
    run_on_host("a0", [], env_for(bridge, "rerun", PRINT_LINES="2"))
    assert len([line for line in capfd.readouterr().err.splitlines() if "] line " in line]) == 2


def _gone(pid: int) -> bool:
    """Dead, or a zombie awaiting its (possibly absent) reaper: either way it no longer runs."""
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False)
    out = out.stdout.strip()
    return out == "" or out.startswith("Z")


def test_stopping_kills_the_whole_process_group():
    from flyte._internal.apple_silicon._daemon import kill_group

    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print(child.pid, flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    grandchild = int(leader.stdout.readline())
    kill_group(leader.pid, grace=10.0, leader=leader)
    assert leader.poll() is not None  # reaped, not left a zombie
    deadline = time.monotonic() + 5
    while not _gone(grandchild) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _gone(grandchild)


# Requests: malformed input is a 4xx with a reason, not a dropped connection.


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"entrypoint": "a0", "argv": "--inputs"}, "argv"),
        ({"entrypoint": "a0", "env": ["A=1"]}, "env"),
        ({"entrypoint": "a0", "env": {"A": 1}}, "env"),
        ({"entrypoint": "a0", "env": {"A": "x\x00y"}}, "NUL"),
        ({"entrypoint": "a0", "python": "3.12; rm -rf /"}, "python"),
        ({"entrypoint": "a0", "pip_args": ["--python", "/usr/bin/python3"]}, "pip option"),
        ({"entrypoint": "sh"}, "entrypoint"),
    ],
)
def test_bad_exec_requests_are_rejected(bridge, spec, message):
    status, body = call(bridge, "POST", "/apple-silicon/v1/workloads/bad/exec", "bad", spec)
    assert status == 400 and message in body["error"]["message"]


def test_malformed_bodies_and_offsets_get_an_answer(bridge):
    assert call(bridge, "POST", "/apple-silicon/v1/workloads/b/exec", "b", raw=b"[1,2]")[0] == 400
    assert call(bridge, "POST", "/apple-silicon/v1/workloads/b/exec", "b", raw=b"{", headers={})[0] == 400
    spec = {"entrypoint": "a0", "argv": [], "env": {}, "packages": []}
    call(bridge, "POST", "/apple-silicon/v1/workloads/b/exec", "b", spec)
    assert call(bridge, "GET", "/apple-silicon/v1/workloads/b/logs?offset=abc", "b")[0] == 400
    assert call(bridge, "GET", "/apple-silicon/v1/workloads/b/logs?offset=-5", "b")[0] == 200


def test_daemon_routes_on_a_host_workload_are_a_conflict(bridge):
    spec = {"entrypoint": "fserve", "argv": [], "env": {}, "packages": []}
    call(bridge, "POST", "/apple-silicon/v1/workloads/h/exec", "h", spec)
    assert call(bridge, "POST", "/apple-silicon/v1/workloads/h/acquire", "h", {})[0] == 409
    assert call(bridge, "POST", "/v1/chat/completions", "h", {})[0] == 409


def test_pip_args_are_limited_to_index_options():
    assert validate_pip_args(["--index-url", "https://x/simple", "--prerelease", "allow"])
    with pytest.raises(ValueError):
        validate_pip_args(["-r", "other.txt"])
    with pytest.raises(ValueError):
        validate_pip_args(["torch"])


def test_workload_file_names_are_safe_and_distinct():
    names = {safe_name(i) for i in ("a:b", "a_b", "a/b", "..", ".", "app-x")}
    assert len(names) == 6
    assert all("/" not in n and n not in (".", "..") for n in names)


def test_log_reads_end_on_a_line(tmp_path):
    log = tmp_path / "log"
    log.write_bytes((("é" * 10) + "\n").encode() * 100)
    chunk, nxt = read_log_lines(log, 0, limit=64)
    assert chunk.endswith(b"\n") and nxt == len(chunk)
    chunk.decode()  # never splits a multi-byte character


# The secret and the bridge process.


def test_an_empty_secret_file_is_replaced_not_trusted(tmp_path):
    (tmp_path / "secret").write_text("")
    secret = _host.load_or_create_secret(tmp_path)
    assert len(secret) >= 32
    assert (tmp_path / "secret").stat().st_mode & 0o777 == 0o600
    assert _host.load_or_create_secret(tmp_path) == secret


def test_admin_route_refuses_a_bridge_without_a_secret(tmp_path):
    b = Bridge("", None, tmp_path / "logs", host_platform="darwin/arm64")
    server = BridgeServer(("127.0.0.1", 0), b)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/apple-silicon/v1/workloads")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)
        assert e.value.code == 401
    finally:
        server.shutdown()


def test_a_stale_pid_file_never_signals_another_process(tmp_path):
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "bridge.json").write_text(json.dumps({"pid": bystander.pid, "port": 1}))
        assert _host.stop_bridge(tmp_path, timeout=1) is False
        assert bystander.poll() is None  # still alive
        assert not (tmp_path / "bridge.json").exists()
    finally:
        bystander.kill()


def test_a_pid_owned_by_root_does_not_crash_stop(tmp_path):
    (tmp_path / "bridge.json").write_text(json.dumps({"pid": 1, "port": 1}))
    assert _host.stop_bridge(tmp_path, timeout=1) is False


def test_status_is_read_only_and_tolerates_odd_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(_host, "running_bridge", lambda state_dir: _host.BridgeProcess(pid=1, port=1))
    _host.bridge_status(tmp_path)
    assert not (tmp_path / "secret").exists()  # `flyte get devbox` must not create state

    (tmp_path / "secret").write_text("s" * 64)

    class Resp:
        status = 200

        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    for body in (b"[1]", b'{"workloads": null}', b'{"workloads": [1, {"workload": "w"}]}'):
        monkeypatch.setattr(_host.urllib.request, "urlopen", lambda req, timeout=None, b=body: Resp(b))
        status = _host.bridge_status(tmp_path)
        assert all(isinstance(w, dict) for w in status["workloads"])


# The CLI.


def test_status_rows_never_interpret_workload_text_as_markup():
    from rich.console import Console

    from flyte.cli._devbox import _apple_silicon_rows

    rows = _apple_silicon_rows(
        {
            "running": True,
            "url": "http://127.0.0.1:30090",
            "pid": 1,
            "workloads": [
                {
                    "workload": "w[bold]",
                    "execution": "host",
                    "entrypoint": "a0",
                    "state": "start_failed",
                    "error_code": "APPLE_SILICON_HOST_EXEC_FAILED",
                    "message": "RuntimeError: [metal::Device] Unable to load kernel [/path]",
                    "log": "/x/[log]",
                },
                "not-a-dict",
            ],
        }
    )
    console = Console(record=True, width=300)
    for key, value in rows:
        console.print(key, value)
    text = console.export_text()
    assert "[metal::Device] Unable to load kernel [/path]" in text
    assert "w[bold]" in text


def test_gpu_and_apple_silicon_flags_conflict():
    import click

    from flyte.cli._devbox import launch_devbox

    with pytest.raises(click.ClickException, match="can't be combined"):
        launch_devbox("img", False, gpu=True, apple_silicon=True)


def test_a_failed_start_stops_the_bridge_it_started(monkeypatch):
    import click

    import flyte.cli._devbox as devbox

    stopped = []
    monkeypatch.setattr(devbox, "_ensure_docker_available", lambda: None)
    monkeypatch.setattr(devbox, "_ensure_volume", lambda name: None)
    monkeypatch.setattr(devbox, "_resolve_apple_silicon", lambda flag, dev: 30090)
    monkeypatch.setattr(devbox, "_container_is_paused", lambda name: False)
    monkeypatch.setattr(devbox, "_container_is_running", lambda name: False)
    monkeypatch.setattr(devbox.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(devbox, "_KUBECONFIG_PATH", Path("/nonexistent/kubeconfig"))
    monkeypatch.setattr(devbox, "_KUBE_DIR", Path("/tmp"))
    monkeypatch.setattr(devbox, "stop_apple_silicon_bridge", lambda: stopped.append(True) or True)

    def boom(*a, **kw):
        raise click.ClickException("port 30080 is already allocated")

    monkeypatch.setattr(devbox, "_launch_devbox_plain", boom)
    with pytest.raises(click.ClickException):
        devbox.launch_devbox("img", False, log_format="json", apple_silicon=True)
    assert stopped == [True]


def test_prefetch_does_not_offer_the_devbox_gpu():
    from flyte.cli._prefetch import ACCELERATOR_CHOICES

    assert not [a for a in ACCELERATOR_CHOICES if a.startswith("AppleSilicon")]


# Serialization.


def test_device_class_must_match_the_apple_silicon_device():
    with pytest.raises(ValueError):
        flyte.Device(quantity=1, device_class="GPU", device="AppleSilicon")
    with pytest.raises(ValueError):
        flyte.Device(quantity=1, device_class="APPLE_SILICON", device="A100")


def test_requirements_files_carry_over_to_the_mac(tmp_path):
    from flyte._internal.runtime.resources_serde import get_apple_silicon_execution_env

    (tmp_path / "base.txt").write_text("numpy==2.1.0 \\\n    --hash=sha256:abc\n")
    (tmp_path / "requirements.txt").write_text(
        "--index-url https://pypi.example.com/simple\n"
        "-r base.txt\n"
        "mlx==0.20.0 \\\n    --hash=sha256:def\n"
        "flyte~=2.0  # the host brings its own\n"
        "torch>=2.4; sys_platform == 'darwin'\n"
    )
    image = (
        flyte.Image.from_debian_base()
        .with_requirements(tmp_path / "requirements.txt")
        .with_pip_packages("transformers", extra_index_urls=["https://extra.example.com/simple"], pre=True)
    )
    env = get_apple_silicon_execution_env(flyte.Resources(gpu="AppleSilicon"), image, None)
    assert json.loads(env["_F_APPLE_SILICON_HOST_PACKAGES"]) == [
        "numpy==2.1.0",
        "mlx==0.20.0",
        "torch>=2.4; sys_platform == 'darwin'",
        "transformers",
    ]
    pip_args = json.loads(env["_F_APPLE_SILICON_HOST_PIP_ARGS"])
    assert pip_args[:2] == ["--index-url", "https://pypi.example.com/simple"]
    assert "https://extra.example.com/simple" in pip_args and "--prerelease" in pip_args
    assert validate_pip_args(pip_args)


def test_a_remote_override_keeps_the_task_on_the_mac():
    from flyte.remote._task import _apple_silicon_env_after_override

    old = {"_F_APPLE_SILICON_EXECUTION": "host", "_F_APPLE_SILICON_HOST_PACKAGES": '["torch"]', "A": "1"}
    carried = _apple_silicon_env_after_override(old, {"B": "2"}, None, "t")
    assert carried == {"_F_APPLE_SILICON_EXECUTION": "host", "_F_APPLE_SILICON_HOST_PACKAGES": '["torch"]'}
    assert _apple_silicon_env_after_override(old, None, flyte.Resources(gpu="A100:1"), "t") == {}
    assert _apple_silicon_env_after_override(old, {"FLYTE_APPLE_SILICON_EXECUTION": "container"}, None, "t") == {}
    with pytest.raises(flyte.errors.RemoteTaskUsageError, match="Deploy it with"):
        _apple_silicon_env_after_override({"A": "1"}, None, flyte.Resources(gpu="AppleSilicon"), "t")


def test_the_container_opt_out_also_works_at_run_time():
    from flyte._internal.apple_silicon._client import host_execution_requested

    env = {"_F_APPLE_SILICON_EXECUTION": "host", "FLYTE_APPLE_SILICON_ENDPOINT": "http://h:1"}
    assert host_execution_requested(env)
    assert not host_execution_requested({**env, "FLYTE_APPLE_SILICON_EXECUTION": "container"})


def test_setup_errors_are_not_retried():
    from flyte._internal.runtime.convert import convert_from_native_to_error

    assert not convert_from_native_to_error(flyte.errors.AppleSiliconNotAvailableError("x")).recoverable
    assert not convert_from_native_to_error(flyte.errors.AppleSiliconHostExecFailedError("x")).recoverable
    assert convert_from_native_to_error(flyte.errors.AppleSiliconBridgeUnreachableError("x")).recoverable
