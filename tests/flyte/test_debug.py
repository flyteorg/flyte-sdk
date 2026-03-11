"""Tests for the debug flag in flyte.run / flyte run and the _debug.client helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from flyteidl2.core.execution_pb2 import TaskLog

from flyte._debug.client import _build_full_url, _extract_vscode_uri
from flyte._run import _Runner

# ---------------------------------------------------------------------------
# _Runner: debug flag sets _F_E_VS env var
# ---------------------------------------------------------------------------


class TestRunnerDebugFlag:
    def test_debug_false_by_default(self):
        runner = _Runner()
        assert runner._debug is False

    def test_debug_true_stored(self):
        runner = _Runner(debug=True)
        assert runner._debug is True


# ---------------------------------------------------------------------------
# with_runcontext passes debug through
# ---------------------------------------------------------------------------


def test_with_runcontext_debug():
    from flyte._run import with_runcontext

    ctx = with_runcontext(mode="local", debug=True)
    assert ctx._debug is True


def test_with_runcontext_debug_default():
    from flyte._run import with_runcontext

    ctx = with_runcontext(mode="local")
    assert ctx._debug is False


# ---------------------------------------------------------------------------
# CLI: --debug flag registered in RunArguments
# ---------------------------------------------------------------------------


def test_run_help_shows_debug_flag():
    from flyte.cli._run import run

    runner = CliRunner()
    result = runner.invoke(run, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--debug" in result.output


def test_run_arguments_debug_default():
    from flyte.cli._run import RunArguments

    args = RunArguments()
    assert args.debug is False


def test_run_arguments_debug_true():
    from flyte.cli._run import RunArguments

    args = RunArguments(debug=True)
    assert args.debug is True


# ---------------------------------------------------------------------------
# _build_full_url
# ---------------------------------------------------------------------------


class TestBuildFullUrl:
    def test_dns_endpoint(self):
        url = _build_full_url("dns:///demo.hosted.unionai.cloud", "/dataplane/pod/v1/foo")
        assert url == "https://demo.hosted.unionai.cloud/dataplane/pod/v1/foo"

    def test_https_endpoint(self):
        url = _build_full_url("https://demo.hosted.unionai.cloud", "/dataplane/pod/v1/foo")
        assert url == "https://demo.hosted.unionai.cloud/dataplane/pod/v1/foo"

    def test_endpoint_with_port(self):
        url = _build_full_url("dns:///demo.hosted.unionai.cloud:443", "/path")
        assert url == "https://demo.hosted.unionai.cloud/path"

    def test_plain_hostname(self):
        url = _build_full_url("my.cluster.example.com", "/path")
        assert url == "https://my.cluster.example.com/path"


# ---------------------------------------------------------------------------
# _extract_vscode_uri
# ---------------------------------------------------------------------------


def _make_action_details(attempts):
    """Build a minimal ActionDetails-like object for testing."""
    details = MagicMock()
    details.pb2.attempts = attempts
    return details


def _make_attempt(logs=None):
    attempt = SimpleNamespace()
    attempt.log_info = logs or []
    return attempt


def _make_log(uri, *, ready=False, link_type=TaskLog.LinkType.EXTERNAL):
    return SimpleNamespace(uri=uri, ready=ready, link_type=link_type)


class TestExtractVscodeUri:
    def test_returns_none_when_no_attempts(self):
        details = _make_action_details([])
        assert _extract_vscode_uri(details) is None

    def test_returns_none_when_no_matching_log(self):
        attempt = _make_attempt(
            logs=[_make_log("https://example.com/logs", ready=True, link_type=TaskLog.LinkType.EXTERNAL)],
        )
        details = _make_action_details([attempt])
        assert _extract_vscode_uri(details) is None

    def test_returns_none_when_ide_log_not_ready(self):
        attempt = _make_attempt(
            logs=[_make_log("/dataplane/pod/v1/foo", ready=False, link_type=TaskLog.LinkType.IDE)],
        )
        details = _make_action_details([attempt])
        assert _extract_vscode_uri(details) is None

    def test_returns_none_when_ready_but_wrong_link_type(self):
        attempt = _make_attempt(
            logs=[_make_log("/dataplane/pod/v1/foo", ready=True, link_type=TaskLog.LinkType.DASHBOARD)],
        )
        details = _make_action_details([attempt])
        assert _extract_vscode_uri(details) is None

    def test_returns_uri_when_ready_and_ide_type(self):
        attempt = _make_attempt(
            logs=[
                _make_log("https://example.com/logs", ready=True, link_type=TaskLog.LinkType.EXTERNAL),
                _make_log("/dataplane/pod/v1/foo", ready=True, link_type=TaskLog.LinkType.IDE),
            ],
        )
        details = _make_action_details([attempt])
        assert _extract_vscode_uri(details) == "/dataplane/pod/v1/foo"

    def test_skips_not_ready_log_picks_ready_one(self):
        attempt = _make_attempt(
            logs=[
                _make_log("/wrong", ready=False, link_type=TaskLog.LinkType.IDE),
                _make_log("/correct", ready=True, link_type=TaskLog.LinkType.IDE),
            ],
        )
        details = _make_action_details([attempt])
        assert _extract_vscode_uri(details) == "/correct"


# ---------------------------------------------------------------------------
# Run.get_debug_url
# ---------------------------------------------------------------------------


class TestRunGetDebugUrl:
    def _make_run(self):
        from flyte.remote._run import Run

        mock_run_pb2 = MagicMock()
        mock_run_pb2.HasField.return_value = True
        return Run(pb2=mock_run_pb2)

    def test_returns_url_when_ready(self):
        run = self._make_run()
        with patch(
            "flyte._debug.client.watch_for_vscode_url",
            return_value="https://demo.hosted.unionai.cloud/dataplane/pod/v1/foo",
        ):
            assert run.get_debug_url() == "https://demo.hosted.unionai.cloud/dataplane/pod/v1/foo"

    def test_returns_none_when_not_found(self):
        run = self._make_run()
        with patch("flyte._debug.client.watch_for_vscode_url", return_value=None):
            assert run.get_debug_url() is None

    def test_caches_result(self):
        run = self._make_run()
        expected = "https://demo.hosted.unionai.cloud/dataplane/pod/v1/foo"
        with patch("flyte._debug.client.watch_for_vscode_url", return_value=expected) as mock_watch:
            assert run.get_debug_url() == expected
            assert run.get_debug_url() == expected
            mock_watch.assert_called_once()
