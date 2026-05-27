"""Tests for `flyte edit settings` — error-header helpers and the
non-interactive ``--from-file`` path."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from flyte.cli._edit import (
    _EDIT_ERROR_END,
    _EDIT_ERROR_START,
    _prepend_error_header,
    _strip_error_header,
    edit,
)


def test_strip_error_header_returns_content_unchanged_when_absent():
    content = "run.default_queue: gpu\n"
    assert _strip_error_header(content) == content


def test_strip_error_header_removes_block(tmp_path: Path):
    body = "run.default_queue: gpu\n"
    content = f"{_EDIT_ERROR_START}\n## bad yaml\n{_EDIT_ERROR_END}\n\n{body}"
    assert _strip_error_header(content) == body


def test_prepend_error_header_replaces_previous(tmp_path: Path):
    path = tmp_path / "buffer.yaml"
    path.write_text("run.default_queue: gpu\n")

    _prepend_error_header(path, "first parse failure")
    first = path.read_text()
    assert first.startswith(_EDIT_ERROR_START)
    assert "## first parse failure" in first
    assert "run.default_queue: gpu" in first

    # Second failure must not stack — user only sees the latest error block.
    _prepend_error_header(path, "second parse failure")
    second = path.read_text()
    assert second.count(_EDIT_ERROR_START) == 1
    assert "## second parse failure" in second
    assert "## first parse failure" not in second
    assert "run.default_queue: gpu" in second


def test_prepend_error_header_preserves_body_exactly(tmp_path: Path):
    path = tmp_path / "buffer.yaml"
    body = "### Settings for scope: ORG\n\nrun.default_queue: gpu\n"
    path.write_text(body)

    _prepend_error_header(path, "boom")
    restored = _strip_error_header(path.read_text())
    assert restored == body


# ---------------------------------------------------------------------------
# `flyte edit settings --from-file` — non-interactive path
# ---------------------------------------------------------------------------


def _fake_settings(local_overrides: dict | None = None) -> MagicMock:
    """Build a mock Settings object matching the surface used by the command."""
    s = MagicMock()
    s.scope_description.return_value = "DOMAIN(prod)"
    s.local_overrides.return_value = local_overrides or {}
    s.update_settings = MagicMock()
    return s


def _invoke_from_file(body: str, *, tmp_path: Path, local_overrides: dict | None = None) -> tuple:
    """Invoke ``flyte edit settings --from-file=<path>`` against mocked remote
    plumbing. Returns (result, settings_mock, cfg_mock)."""
    yaml_file = tmp_path / "overrides.yaml"
    yaml_file.write_text(body)

    import logging

    settings_mock = _fake_settings(local_overrides=local_overrides)
    cfg_mock = MagicMock()
    # CommandBase.invoke reads cfg.log_level and compares to logging.DEBUG when
    # an exception escapes — must be an actual int to survive the comparison.
    cfg_mock.log_level = logging.INFO

    with (
        patch("flyte.remote.Settings.get_settings_for_edit", return_value=settings_mock),
        patch("flyte.cli._edit.subprocess.run") as run_mock,  # guard: editor must not be invoked
    ):
        runner = CliRunner()
        result = runner.invoke(
            edit,
            ["settings", "--from-file", str(yaml_file), "--domain", "prod"],
            obj=cfg_mock,
            catch_exceptions=False,
        )

    assert not run_mock.called, "--from-file must never launch an editor"
    return result, settings_mock, cfg_mock


def test_from_file_applies_overrides(tmp_path: Path):
    result, s, _ = _invoke_from_file("run.default_queue: gpu\nsecurity.service_account: ml-sa\n", tmp_path=tmp_path)
    assert result.exit_code == 0, result.output
    s.update_settings.assert_called_once_with({"run.default_queue": "gpu", "security.service_account": "ml-sa"})
    assert "Settings updated successfully" in result.output


def test_from_file_no_changes(tmp_path: Path):
    """When the file's overrides match the current local state, we should
    skip the RPC and report no changes."""
    result, s, _ = _invoke_from_file(
        "run.default_queue: gpu\n",
        tmp_path=tmp_path,
        local_overrides={"run.default_queue": "gpu"},
    )
    assert result.exit_code == 0, result.output
    s.update_settings.assert_not_called()
    assert "No changes detected" in result.output


def test_from_file_parse_error_aborts(tmp_path: Path):
    """An invalid YAML payload must abort without invoking update_settings."""
    result, s, _ = _invoke_from_file("not a mapping\n- just\n- a\n- list\n", tmp_path=tmp_path)
    assert result.exit_code != 0
    s.update_settings.assert_not_called()
    assert "Invalid YAML" in result.output


def test_from_file_handles_commented_template(tmp_path: Path):
    """A file pulled from `flyte get settings` (with ### / ## / # markers)
    parses via yaml.safe_load — only uncommented keys become overrides."""
    body = (
        "### Settings for scope: DOMAIN(prod)\n"
        "\n"
        "### Local overrides\n"
        "## Kubernetes service account for task pods\n"
        "security.service_account: ml-sa\n"
        "\n"
        "### Available settings (uncomment and edit to set at this scope)\n"
        "## Default queue for task runs\n"
        "# run.default_queue: ''\n"
    )
    result, s, _ = _invoke_from_file(body, tmp_path=tmp_path)
    assert result.exit_code == 0, result.output
    s.update_settings.assert_called_once_with({"security.service_account": "ml-sa"})


# ---------------------------------------------------------------------------
# `flyte get settings --to-file` — dump to disk
# ---------------------------------------------------------------------------


def test_get_settings_to_file_roundtrips_through_from_file(tmp_path: Path):
    """`get --to-file` must write YAML that `edit --from-file` re-parses."""
    import logging

    from flyte.cli._get import get
    from flyte.remote._settings import (
        EffectiveSetting,
        LocalSetting,
        SettingOrigin,
        Settings,
    )

    out_path = tmp_path / "settings.yaml"

    # Populate a real Settings object so to_yaml() renders actual content.
    real_settings = Settings(
        effective_settings=[
            EffectiveSetting(key="run.default_queue", value="gpu", origin=SettingOrigin("DOMAIN", "prod")),
        ],
        local_settings=[LocalSetting(key="security.service_account", value="ml-sa")],
        domain="prod",
        project="ml",
        _version=7,
    )

    cfg_mock = MagicMock()
    cfg_mock.log_level = logging.INFO

    with patch("flyte.remote.Settings.get_settings_for_edit", return_value=real_settings):
        runner = CliRunner()
        result = runner.invoke(
            get,
            ["settings", "--domain", "prod", "--project", "ml", "--to-file", str(out_path)],
            obj=cfg_mock,
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    dumped = out_path.read_text()
    # The comment-tier markers must survive to disk (so the file is editable).
    assert "### Settings for scope: PROJECT(prod/ml)" in dumped
    assert "### Local overrides" in dumped
    assert "security.service_account: ml-sa" in dumped

    # And `edit --from-file` parses the same file without stripping live overrides.
    assert Settings.parse_yaml(dumped) == {"security.service_account": "ml-sa"}
