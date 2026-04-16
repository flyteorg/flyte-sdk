"""Tests for the error-header helpers used by `flyte edit settings`."""

from pathlib import Path

from flyte.cli._edit import (
    _EDIT_ERROR_END,
    _EDIT_ERROR_START,
    _prepend_error_header,
    _strip_error_header,
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
