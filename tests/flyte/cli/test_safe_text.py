"""Tests for ASCII-safe console text on non-UTF stdout encodings.

Legacy Windows consoles run on a regional code page with no room for emoji, so a
decorative glyph reaches Rich's renderer and raises UnicodeEncodeError there, taking
the command down (FLYTE-SDK-7M, a `flyte start devbox` on a cp936 console).
"""

import io
from unittest import mock

import pytest
from rich.console import Console
from rich.panel import Panel

from flyte._code_bundle._utils import HOME_DIRECTORY_WARNING
from flyte.cli._common import print_url, safe_text

DEVBOX_READY_PANEL = (
    "[green bold]Flyte devbox cluster is ready![/green bold]\n\n"
    "  \U0001f680 UI:             [link=http://localhost:30080/v2]http://localhost:30080/v2[/link]\n"
    "  \U0001f433 Image Registry: localhost:30000"
)


def _stdout_with_encoding(encoding: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding)


def _console_on(encoding: str) -> tuple[Console, io.TextIOWrapper]:
    """A Console writing to a strict stream on `encoding`, like a legacy Windows console."""
    stream = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")
    return Console(file=stream, width=100, color_system=None), stream


def test_utf8_stdout_is_left_alone():
    with mock.patch("sys.stdout", _stdout_with_encoding("utf-8")):
        assert safe_text(DEVBOX_READY_PANEL) == DEVBOX_READY_PANEL


def test_ascii_only_text_is_left_alone():
    with mock.patch("sys.stdout", _stdout_with_encoding("cp936")):
        assert safe_text("Flyte devbox cluster is ready!") == "Flyte devbox cluster is ready!"


@pytest.mark.parametrize("encoding", ["cp936", "cp1252", "ascii"])
def test_emoji_dropped_on_legacy_code_pages(encoding: str):
    with mock.patch("sys.stdout", _stdout_with_encoding(encoding)):
        out = safe_text(DEVBOX_READY_PANEL)
    assert "\U0001f680" not in out
    assert "\U0001f433" not in out
    # Only the ornament goes; the message and its Rich markup survive intact.
    assert "Flyte devbox cluster is ready!" in out
    assert "[link=http://localhost:30080/v2]" in out
    assert "Image Registry: localhost:30000" in out
    out.encode(encoding)  # would raise if anything unencodable were left


def test_meaningful_glyphs_keep_an_ascii_stand_in():
    with mock.patch("sys.stdout", _stdout_with_encoding("cp936")):
        assert safe_text("➡️  ") == "->  "
        assert safe_text("⚠️ warning").startswith("! ")


def test_unmapped_unencodable_character_becomes_a_question_mark():
    # Not in the fallback table: it must still not reach the stream unencoded.
    with mock.patch("sys.stdout", _stdout_with_encoding("cp1252")):
        assert safe_text("run 中文 done") == "run ?? done"


def test_unknown_encoding_name_degrades_to_ascii():
    class _BogusEncoding:
        encoding = "not-a-real-codec"

    with mock.patch("sys.stdout", _BogusEncoding()):
        assert safe_text("\U0001f680 UI") == " UI"


def test_missing_encoding_attribute_degrades_to_ascii():
    class _NoEncoding:
        encoding = None

    with mock.patch("sys.stdout", _NoEncoding()):
        assert safe_text("\U0001f680 UI") == " UI"


def test_devbox_ready_panel_renders_on_a_cp936_console():
    """The FLYTE-SDK-7M crash, end to end: Rich writing the panel to a cp936 stream."""
    console, stream = _console_on("cp936")
    with mock.patch("sys.stdout", _stdout_with_encoding("cp936")):
        body = safe_text(DEVBOX_READY_PANEL)
    console.print(Panel(body, title="[bold]Flyte Devbox[/bold]", border_style="green"))
    stream.flush()
    rendered = stream.buffer.getvalue().decode("cp936")
    assert "Flyte devbox cluster is ready!" in rendered


def test_print_url_prefix_renders_on_a_cp936_console():
    """`flyte run` prints the run URL behind a default arrow-emoji prefix."""
    console, stream = _console_on("cp936")
    with mock.patch("sys.stdout", _stdout_with_encoding("cp936")):
        print_url(console, "https://example.com/run/abc")
    stream.flush()
    rendered = stream.buffer.getvalue().decode("cp936")
    assert "https://example.com/run/abc" in rendered
    assert rendered.startswith("->")


def test_home_directory_warning_renders_on_a_cp936_console():
    console, stream = _console_on("cp936")
    with mock.patch("sys.stdout", _stdout_with_encoding("cp936")):
        warning = safe_text(HOME_DIRECTORY_WARNING.format(path="/home/u"))
    console.print(f"[yellow]Warning: {warning}[/yellow]")
    stream.flush()
    rendered = stream.buffer.getvalue().decode("cp936")
    assert "Running from your home directory" in rendered
