"""Unit tests for the `flyte-rapidmlx` preflight wrapper (`_cli.py`)."""

import sys

import pytest

from flyteplugins.rapidmlx import _cli

# ---------------------------------------------------------------------------
# resolve_rapidmlx
# ---------------------------------------------------------------------------


def test_resolve_rapidmlx_via_path(monkeypatch, tmp_path):
    fake = tmp_path / "rapidmlx"
    fake.write_bytes(b"#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(_cli.shutil, "which", lambda _name: str(fake))
    assert _cli.resolve_rapidmlx() == str(fake)


def test_resolve_rapidmlx_via_homebrew_when_not_on_path(monkeypatch, tmp_path):
    """Host-execution PATH omits Homebrew, so resolve_rapidmlx must probe it directly."""
    monkeypatch.setattr(_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_cli, "_HOMEBREW_RAPID_MLX_CANDIDATES", (str(tmp_path / "bin" / "rapidmlx"),))
    fake = tmp_path / "bin" / "rapidmlx"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"#!/bin/sh\n")
    fake.chmod(0o755)
    assert _cli.resolve_rapidmlx() == str(fake)


def test_resolve_rapidmlx_nonexecutable_homebrew_skipped(monkeypatch, tmp_path):
    """A Homebrew rapidmlx that isn't a file (or isn't executable) is not a match."""
    monkeypatch.setattr(_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_cli, "_HOMEBREW_RAPID_MLX_CANDIDATES", (str(tmp_path / "missing" / "rapidmlx"),))
    assert _cli.resolve_rapidmlx() is None


def test_resolve_rapidmlx_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_cli, "_HOMEBREW_RAPID_MLX_CANDIDATES", ())
    assert _cli.resolve_rapidmlx() is None


def test_resolve_rapidmlx_checks_homebrew_after_path(monkeypatch, tmp_path):
    """An rapidmlx on PATH wins over a Homebrew candidate."""
    on_path = tmp_path / "path" / "rapidmlx"
    on_path.parent.mkdir(parents=True)
    on_path.write_bytes(b"#!/bin/sh\n")
    on_path.chmod(0o755)
    monkeypatch.setattr(_cli.shutil, "which", lambda _name: str(on_path))
    monkeypatch.setattr(_cli, "_HOMEBREW_RAPID_MLX_CANDIDATES", (str(tmp_path / "brew" / "rapidmlx"),))
    assert _cli.resolve_rapidmlx() == str(on_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _make_binary(tmp_path):
    binary = tmp_path / "rapidmlx"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    return binary


def test_main_execs_real_binary(monkeypatch, tmp_path, capsys):
    binary = _make_binary(tmp_path)
    monkeypatch.setattr(_cli, "resolve_rapidmlx", lambda: str(binary))
    argv = ["flyte-rapidmlx", "serve", "--port", "8000"]
    monkeypatch.setattr(sys, "argv", argv)

    captured_exec: dict = {}

    def fake_execv(path, args):
        captured_exec["path"] = path
        captured_exec["args"] = args
        raise SystemExit(0)  # like a real execv that never returns

    monkeypatch.setattr(_cli.os, "execv", fake_execv)
    with pytest.raises(SystemExit):
        _cli.main()
    assert captured_exec["path"] == str(binary)
    assert captured_exec["args"] == [str(binary), "serve", "--port", "8000"]


def test_main_prints_install_hint_when_missing(monkeypatch, capsys):
    monkeypatch.setattr(_cli, "resolve_rapidmlx", lambda: None)

    rc = _cli.main()
    assert rc == 127
    out = capsys.readouterr()
    assert "Rapid-MLX is not installed" in out.err


def test_main_argument_order_preserves_argv_after_binary(monkeypatch, tmp_path):
    binary = _make_binary(tmp_path)
    monkeypatch.setattr(_cli, "resolve_rapidmlx", lambda: str(binary))
    monkeypatch.setattr(sys, "argv", ["flyte-rapidmlx", "serve", "/a b"])

    captured_exec: dict = {}

    def fake_execv(path, args):
        captured_exec["args"] = args
        raise SystemExit(0)

    monkeypatch.setattr(_cli.os, "execv", fake_execv)
    with pytest.raises(SystemExit):
        _cli.main()
    assert captured_exec["args"] == [str(binary), "serve", "/a b"]
