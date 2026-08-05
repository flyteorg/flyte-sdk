from flyte.cli._start import start
from flyte.cli._tui import config_is_remote


def test_start_includes_tui_command():
    assert "tui" in start.commands


def test_start_has_no_remote_tui_command():
    # remote-tui was unified into `flyte start tui`; the separate command is gone.
    assert "remote-tui" not in start.commands


def test_tui_command_has_config_option():
    tui = start.commands["tui"]
    assert "--config" in tui.params[0].opts or any("--config" in p.opts for p in tui.params)


def test_config_is_remote_detects_endpoint(tmp_path):
    remote_cfg = tmp_path / "remote.yaml"
    remote_cfg.write_text(
        "admin:\n  endpoint: dns:///example.flyte.org\ntask:\n  domain: development\n",
        encoding="utf-8",
    )
    assert config_is_remote(str(remote_cfg)) is True


def test_config_is_remote_false_for_local(tmp_path, monkeypatch):
    monkeypatch.delenv("FLYTE_API_KEY", raising=False)
    local_cfg = tmp_path / "local.yaml"
    local_cfg.write_text("local:\n  persistence: true\n", encoding="utf-8")
    assert config_is_remote(str(local_cfg)) is False


def test_config_is_remote_true_when_api_key_set(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYTE_API_KEY", "some-key")
    local_cfg = tmp_path / "local.yaml"
    local_cfg.write_text("local:\n  persistence: true\n", encoding="utf-8")
    assert config_is_remote(str(local_cfg)) is True
