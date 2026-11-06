import json
import sys

import pytest

from flyte._internal.apple_silicon import _bridge, _host
from flyte._internal.apple_silicon._bridge import Bridge
from flyte._internal.apple_silicon._hostexec import DEVBOX_ENDPOINT_REWRITES

SECRET = "test-secret-" + "0" * 40
ONEBOX = {"onebox.onebox.svc:8082": "127.0.0.1:30080"}


def test_bridge_rewrites_default_to_the_devbox(tmp_path):
    assert Bridge(SECRET, None, tmp_path / "logs", host_platform="darwin/arm64").rewrites == DEVBOX_ENDPOINT_REWRITES


def test_bridge_takes_its_own_rewrites(tmp_path):
    b = Bridge(SECRET, None, tmp_path / "logs", host_platform="darwin/arm64", rewrites=ONEBOX)
    assert b.rewrites == ONEBOX
    # An empty map means no rewrites at all, not the devbox's.
    assert Bridge(SECRET, None, tmp_path / "logs", host_platform="darwin/arm64", rewrites={}).rewrites == {}


def _run_main(monkeypatch, tmp_path, *args):
    started = {}
    monkeypatch.setattr(_host, "apple_silicon_unavailable_reason", lambda: None)
    monkeypatch.setattr(_host, "resolve_daemon_command", lambda: None)
    monkeypatch.setattr(_bridge, "serve", lambda bridge, host, port: started.update(bridge=bridge))
    assert _bridge.main(["--state-dir", str(tmp_path), *args]) == 0
    return started["bridge"]


def test_main_parses_rewrites(monkeypatch, tmp_path):
    b = _run_main(monkeypatch, tmp_path, "--rewrite", "onebox.onebox.svc:8082=127.0.0.1:30080", "--rewrite", "a:1=b:2")
    assert b.rewrites == {"onebox.onebox.svc:8082": "127.0.0.1:30080", "a:1": "b:2"}
    assert _run_main(monkeypatch, tmp_path).rewrites == DEVBOX_ENDPOINT_REWRITES
    # No rewrites at all, rather than the devbox's.
    assert _run_main(monkeypatch, tmp_path, "--no-default-rewrites").rewrites == {}


def test_main_refuses_a_malformed_rewrite(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, tmp_path, "--rewrite", "no-equals-sign")
    assert "IN_CLUSTER=ON_HOST" in capsys.readouterr().err


def test_the_recorded_bridge_keeps_its_rewrites(tmp_path):
    (tmp_path / "bridge.json").write_text(json.dumps({"pid": 1, "port": 30091, "rewrites": ONEBOX}))
    assert _host._read_process(tmp_path).rewrites == ONEBOX
    (tmp_path / "bridge.json").write_text(json.dumps({"pid": 1, "port": 30090}))
    assert _host._read_process(tmp_path).rewrites is None


def test_start_bridge_passes_and_records_rewrites(monkeypatch, tmp_path):
    calls = {}

    class Proc:
        pid = 4242

        def poll(self):
            return None

    def popen(cmd, **kwargs):
        calls["cmd"] = cmd
        return Proc()

    monkeypatch.setattr(_host, "apple_silicon_unavailable_reason", lambda: None)
    monkeypatch.setattr(_host.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(_host, "running_bridge", lambda state_dir=None: None)
    monkeypatch.setattr(_host, "bridge_healthy", lambda port, timeout=2.0: "cmd" in calls)
    monkeypatch.setattr(_host.subprocess, "Popen", popen)

    proc = _host.start_bridge(30091, state_dir=tmp_path, rewrites=ONEBOX)
    assert proc.rewrites == ONEBOX
    cmd = calls["cmd"]
    assert "--no-default-rewrites" in cmd
    assert cmd[cmd.index("--rewrite") + 1] == "onebox.onebox.svc:8082=127.0.0.1:30080"
    assert json.loads((tmp_path / "bridge.json").read_text())["rewrites"] == ONEBOX


def test_an_empty_map_reaches_the_bridge_as_no_rewrites(monkeypatch, tmp_path):
    calls = {}

    class Proc:
        pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(_host, "apple_silicon_unavailable_reason", lambda: None)
    monkeypatch.setattr(_host.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(_host, "running_bridge", lambda state_dir=None: None)
    monkeypatch.setattr(_host, "bridge_healthy", lambda port, timeout=2.0: "cmd" in calls)
    monkeypatch.setattr(_host.subprocess, "Popen", lambda cmd, **kw: calls.update(cmd=cmd) or Proc())

    _host.start_bridge(30091, state_dir=tmp_path, rewrites={})
    assert "--no-default-rewrites" in calls["cmd"]
    assert "--rewrite" not in calls["cmd"]
    # Without a map the bridge keeps the devbox's defaults: no flag.
    calls.clear()
    _host.start_bridge(30092, state_dir=tmp_path / "other")
    assert "--no-default-rewrites" not in calls["cmd"]


def test_a_bridge_with_other_rewrites_is_restarted(monkeypatch, tmp_path):
    stopped = []
    monkeypatch.setattr(_host, "apple_silicon_unavailable_reason", lambda: None)
    monkeypatch.setattr(_host.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(_host, "running_bridge", lambda state_dir=None: _host.BridgeProcess(pid=1, port=30091))
    monkeypatch.setattr(_host, "_same_version", lambda proc: True)
    monkeypatch.setattr(_host, "stop_bridge", lambda state_dir=None, timeout=30.0: stopped.append(True) or True)
    launched = []
    monkeypatch.setattr(_host, "bridge_healthy", lambda port, timeout=2.0: bool(launched))

    class Proc:
        pid = 7

        def poll(self):
            return None

    monkeypatch.setattr(_host.subprocess, "Popen", lambda cmd, **kw: launched.append(cmd) or Proc())
    assert _host.start_bridge(30091, state_dir=tmp_path, rewrites=ONEBOX).pid == 7
    assert stopped, "the running bridge had the devbox's rewrites, so it was replaced"
    # The same rewrites reuse it.
    monkeypatch.setattr(
        _host, "running_bridge", lambda state_dir=None: _host.BridgeProcess(pid=1, port=30091, rewrites=ONEBOX)
    )
    assert _host.start_bridge(30091, state_dir=tmp_path, rewrites=ONEBOX).pid == 1


if sys.platform == "win32":  # pragma: no cover
    pytestmark = pytest.mark.skip("the bridge runs on macOS")
