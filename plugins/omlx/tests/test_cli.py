"""Unit tests for the `flyte-omlx` preflight wrapper (`_cli.py`)."""

import sys

import pytest

from flyteplugins.omlx import _cli

# ---------------------------------------------------------------------------
# resolve_omlx
# ---------------------------------------------------------------------------


def test_resolve_omlx_via_path(monkeypatch, tmp_path):
    fake = tmp_path / "omlx"
    fake.write_bytes(b"#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(_cli.shutil, "which", lambda _name: str(fake))
    assert _cli.resolve_omlx() == str(fake)


def test_resolve_omlx_via_homebrew_when_not_on_path(monkeypatch, tmp_path):
    """Host-execution PATH omits Homebrew, so resolve_omlx must probe it directly."""
    monkeypatch.setattr(_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_cli, "_HOMEBREW_OMLX_CANDIDATES", (str(tmp_path / "bin" / "omlx"),))
    fake = tmp_path / "bin" / "omlx"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"#!/bin/sh\n")
    fake.chmod(0o755)
    assert _cli.resolve_omlx() == str(fake)


def test_resolve_omlx_nonexecutable_homebrew_skipped(monkeypatch, tmp_path):
    """A Homebrew omlx that isn't a file (or isn't executable) is not a match."""
    monkeypatch.setattr(_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_cli, "_HOMEBREW_OMLX_CANDIDATES", (str(tmp_path / "missing" / "omlx"),))
    assert _cli.resolve_omlx() is None


def test_resolve_omlx_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_cli, "_HOMEBREW_OMLX_CANDIDATES", ())
    assert _cli.resolve_omlx() is None


def test_resolve_omlx_checks_homebrew_after_path(monkeypatch, tmp_path):
    """An omlx on PATH wins over a Homebrew candidate."""
    on_path = tmp_path / "path" / "omlx"
    on_path.parent.mkdir(parents=True)
    on_path.write_bytes(b"#!/bin/sh\n")
    on_path.chmod(0o755)
    monkeypatch.setattr(_cli.shutil, "which", lambda _name: str(on_path))
    monkeypatch.setattr(_cli, "_HOMEBREW_OMLX_CANDIDATES", (str(tmp_path / "brew" / "omlx"),))
    assert _cli.resolve_omlx() == str(on_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _make_binary(tmp_path):
    binary = tmp_path / "omlx"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    return binary


def test_main_execs_real_binary(monkeypatch, tmp_path, capsys):
    binary = _make_binary(tmp_path)
    monkeypatch.setattr(_cli, "resolve_omlx", lambda: str(binary))
    argv = ["flyte-omlx", "serve", "--port", "8000"]
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


def test_main_falls_back_to_python_package(monkeypatch):
    monkeypatch.setattr(_cli, "resolve_omlx", lambda: None)

    class FakeCli:
        @staticmethod
        def main():
            return 7

    omlx_pkg = type(sys)("omlx")
    omlx_cli_mod = type(sys)("omlx.cli")
    omlx_cli_mod.main = staticmethod(FakeCli.main)
    monkeypatch.setitem(sys.modules, "omlx", omlx_pkg)
    monkeypatch.setitem(sys.modules, "omlx.cli", omlx_cli_mod)
    monkeypatch.setattr(sys, "argv", ["flyte-omlx", "serve"])
    assert _cli.main() == 7


def test_main_prints_install_hint_when_missing(monkeypatch, capsys):
    monkeypatch.setattr(_cli, "resolve_omlx", lambda: None)

    def raise_import_error(*a, **k):
        raise ImportError("no omlx")

    monkeypatch.setattr("builtins.__import__", raise_import_error)
    rc = _cli.main()
    assert rc == 127
    out = capsys.readouterr()
    assert "oMLX is not installed" in out.err


def test_main_argument_order_preserves_argv_after_binary(monkeypatch, tmp_path):
    binary = _make_binary(tmp_path)
    monkeypatch.setattr(_cli, "resolve_omlx", lambda: str(binary))
    monkeypatch.setattr(sys, "argv", ["flyte-omlx", "serve", "--model-dir", "/a b"])

    captured_exec: dict = {}

    def fake_execv(path, args):
        captured_exec["args"] = args
        raise SystemExit(0)

    monkeypatch.setattr(_cli.os, "execv", fake_execv)
    with pytest.raises(SystemExit):
        _cli.main()
    assert captured_exec["args"] == [str(binary), "serve", "--model-dir", "/a b"]
