"""Tests for the CLI's live upload progress panel."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from flyte.cli._progress import _HASH_ROW_MIN_BYTES, _LiveUploadDisplay, upload_display
from flyte.remote import _progress


def _console() -> Console:
    return Console(file=io.StringIO(), width=100, force_terminal=False, no_color=True)


def _render(display: _LiveUploadDisplay) -> str:
    console = _console()
    console.print(display._render())
    return console.file.getvalue()


def _display() -> _LiveUploadDisplay:
    return _LiveUploadDisplay(_console(), "Publishing artifact my_model", "testing/development")


def test_bar_tracks_bytes_and_marks_completion():
    display = _display()
    total = 4 * _HASH_ROW_MIN_BYTES
    display.start("up", name="model.pt", phase="uploading", total=total)
    display.advance("up", total // 2)

    mid = _render(display)
    assert "uploading" in mid
    assert "model.pt" in mid
    assert " 50%" in mid

    display.finish("up")
    done = _render(display)
    assert "uploaded" in done
    assert "100%" in done


def test_tiny_hash_row_is_suppressed_but_upload_row_is_not():
    """A small file hashes instantly; only its upload is worth a row."""
    display = _display()
    display.start("hash", name="card.html", phase="hashing", total=1024)
    display.advance("hash", 1024)
    display.finish("hash")
    assert "hashing" not in _render(display)
    assert "hashed" not in _render(display)

    display.start("up", name="card.html", phase="uploading", total=1024)
    assert "uploading" in _render(display)


def test_large_hash_gets_its_own_row():
    display = _display()
    display.start("hash", name="model.pt", phase="hashing", total=_HASH_ROW_MIN_BYTES + 1)
    assert "hashing" in _render(display)


def test_restart_reuses_the_same_row():
    display = _display()
    total = 4 * _HASH_ROW_MIN_BYTES
    display.start("up", name="model.pt", phase="uploading", total=total)
    display.advance("up", total)
    display.start("up", name="model.pt", phase="uploading", total=total)

    frame = _render(display)
    assert frame.count("model.pt") == 1, "a retry must not stack a second bar"
    assert "  0%" in frame, "a retry restarts from zero"


def test_failure_is_marked_in_place():
    display = _display()
    total = 4 * _HASH_ROW_MIN_BYTES
    display.start("up", name="model.pt", phase="uploading", total=total)
    display.advance("up", total // 4)
    display.finish("up", failed=True)

    frame = _render(display)
    assert "failed" in frame
    assert " 25%" in frame, "the bar stays where it stopped"


def test_unknown_total_still_finishes():
    display = _display()
    display.start("up", name="model.pt", phase="uploading", total=0)
    display.advance("up", 512)
    display.finish("up")
    assert "100%" in _render(display)


def test_note_and_unknown_keys_are_harmless():
    display = _display()
    display.note("registering artifact")
    assert "registering artifact" in _render(display)
    # Events for a key that was never started must not raise.
    display.advance("nope", 1)
    display.finish("nope")


def test_display_installs_and_removes_the_handler(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    console = _console()
    with upload_display("title", console=console) as display:
        assert _progress.current_handler() is display
    assert _progress.current_handler() is None


def test_non_tty_falls_back_to_plain_text(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)
    console = _console()
    with upload_display("Publishing artifact my_model", console=console) as display:
        display.note("uploading html card")
        # No handler is installed, so nothing tries to animate into a log file.
        assert _progress.current_handler() is None

    output = console.file.getvalue()
    assert "Publishing artifact my_model" in output
    assert "uploading html card" in output


def test_no_progress_flag_forces_plain_text(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    console = _console()
    with upload_display("title", no_progress=True, console=console):
        assert _progress.current_handler() is None


@pytest.mark.parametrize("phase", ["hashing", "uploading"])
def test_phases_render_their_verb(phase: str):
    display = _display()
    display.start("k", name="model.pt", phase=phase, total=4 * _HASH_ROW_MIN_BYTES)
    assert phase in _render(display)
