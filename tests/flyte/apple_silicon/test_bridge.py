import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import flyte.errors
from flyte._internal.apple_silicon._bridge import Bridge, BridgeServer
from flyte._internal.apple_silicon._client import attach_from_env
from flyte._internal.apple_silicon._daemon import DaemonState
from flyte._internal.apple_silicon._token import mint_token

SECRET = "test-secret-" + "0" * 40
FAKE_SERVER = [sys.executable, str(Path(__file__).parent / "fake_daemon_server.py")]


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DAEMON_MODE", "ok")
    b = Bridge(SECRET, FAKE_SERVER, tmp_path / "logs", host_platform="darwin/arm64", start_timeout=20)
    server = BridgeServer(("127.0.0.1", 0), b)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    b.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield b
    server.shutdown()
    b.shutdown()


def env_for(bridge, workload, model="some/model"):
    env = {
        "FLYTE_APPLE_SILICON_ENDPOINT": bridge.url,
        "FLYTE_APPLE_SILICON_WORKLOAD": workload,
        "FLYTE_APPLE_SILICON_TOKEN": mint_token(SECRET, workload),
    }
    if model:
        env["FLYTE_APPLE_SILICON_MODEL"] = model
    return env


def call(bridge, method, path, token=None, body=None):
    req = urllib.request.Request(
        f"{bridge.url}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token}"} if token else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_attach_serves_and_proxies(bridge):
    attachment = attach_from_env(env_for(bridge, "wl-ok"))
    assert attachment is not None
    try:
        assert attachment.snapshot["state"] == DaemonState.SERVING.value
        token = mint_token(SECRET, "wl-ok")

        status, body = call(bridge, "POST", "/v1/chat/completions", token, {"messages": []})
        assert status == 200
        payload = json.loads(body)
        assert payload["model"] == "some/model"
        # The workload's credential stops at the bridge.
        assert payload["auth"] is None

        status, body = call(bridge, "POST", "/v1/chat/completions", token, {"messages": [], "stream": True})
        assert status == 200
        assert body.decode().count("data: ") == 4 and body.decode().endswith("data: [DONE]\n\n")

        status, body = call(bridge, "GET", "/apple-silicon/v1/workloads/wl-ok/logs?offset=0", token)
        assert status == 200 and "fake mlx_lm.server starting" in json.loads(body)["data"]
    finally:
        attachment.close()
    assert bridge.get("wl-ok") is None
    assert bridge.snapshot()[0]["state"] == DaemonState.TORN_DOWN.value


def test_tokens_are_scoped_to_their_workload(bridge):
    status, body = call(bridge, "POST", "/v1/chat/completions", "as1.wl-a.forged", {})
    assert status == 401
    assert json.loads(body)["error"]["code"] == "UNAUTHORIZED"

    status, _ = call(bridge, "POST", "/apple-silicon/v1/workloads/wl-b/acquire", mint_token(SECRET, "wl-a"), {})
    assert status == 403

    status, _ = call(bridge, "GET", "/apple-silicon/v1/workloads", mint_token(SECRET, "wl-a"))
    assert status == 401
    status, body = call(bridge, "GET", "/apple-silicon/v1/workloads", SECRET)
    assert status == 200 and json.loads(body) == {"workloads": []}


def test_plain_openai_client_gets_a_daemon_lazily(bridge):
    status, _ = call(bridge, "POST", "/v1/completions", mint_token(SECRET, "wl-lazy"), {"prompt": "x"})
    assert status == 200
    w = bridge.get("wl-lazy")
    assert w is not None and not w.attached


def test_missing_model_is_named(bridge, monkeypatch):
    monkeypatch.setenv("FAKE_DAEMON_MODE", "model_not_found")
    with pytest.raises(flyte.errors.AppleSiliconModelNotFoundError, match="Model 'nope/nope' is not available locally"):
        attach_from_env(env_for(bridge, "wl-missing", model="nope/nope"))


def test_daemon_that_cannot_start_is_named(bridge, monkeypatch):
    monkeypatch.setenv("FAKE_DAEMON_MODE", "crash")
    with pytest.raises(flyte.errors.AppleSiliconDaemonStartFailedError, match="Unable to load kernel"):
        attach_from_env(env_for(bridge, "wl-crash"))


def test_oom_kill_is_surfaced_not_silent(bridge, monkeypatch):
    monkeypatch.setenv("FAKE_DAEMON_MODE", "oom")
    attachment = attach_from_env(env_for(bridge, "wl-oom"))
    token = mint_token(SECRET, "wl-oom")
    try:
        status, _ = call(bridge, "POST", "/v1/completions", token, {"prompt": "big"})
        assert status >= 500  # the daemon died mid-request
        status, body = call(bridge, "POST", "/v1/completions", token, {"prompt": "again"})
        assert status == 503
        error = json.loads(body)["error"]
        assert error["code"] == "APPLE_SILICON_DAEMON_OOM"
        assert "out of memory" in error["message"]
        assert bridge.get("wl-oom").daemon.state == DaemonState.OOM_KILLED
    finally:
        attachment.close()


def test_attach_after_failure_starts_a_fresh_daemon(bridge, monkeypatch):
    monkeypatch.setenv("FAKE_DAEMON_MODE", "crash")
    with pytest.raises(flyte.errors.AppleSiliconDaemonStartFailedError):
        attach_from_env(env_for(bridge, "wl-retry"))
    monkeypatch.setenv("FAKE_DAEMON_MODE", "ok")
    attachment = attach_from_env(env_for(bridge, "wl-retry"))
    assert attachment.snapshot["state"] == "serving"
    attachment.close()


def test_lapsed_lease_tears_the_daemon_down(bridge):
    attachment = attach_from_env(env_for(bridge, "wl-lease"))
    attachment._stop.set()  # the workload's container died: no more heartbeats
    bridge.lease_ttl = 0.1
    time.sleep(0.2)
    bridge.reap()
    assert bridge.get("wl-lease") is None


def test_unreachable_bridge_is_named():
    env = {
        "FLYTE_APPLE_SILICON_ENDPOINT": "http://127.0.0.1:9",
        "FLYTE_APPLE_SILICON_WORKLOAD": "w",
        "FLYTE_APPLE_SILICON_TOKEN": "t",
    }
    with pytest.raises(
        flyte.errors.AppleSiliconBridgeUnreachableError, match="could not reach the Apple Silicon endpoint"
    ):
        attach_from_env(env, start_timeout=1)


def test_not_an_apple_silicon_workload():
    assert attach_from_env({}) is None
