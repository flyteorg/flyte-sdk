"""Tests for ASCII-safe spinner fallback on non-UTF stdout encodings."""

import io
from unittest import mock

from flyte.cli._common import safe_spinner


def _stdout_with_encoding(encoding: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding)


def test_safe_spinner_keeps_dots_on_utf8():
    with mock.patch("sys.stdout", _stdout_with_encoding("utf-8")):
        assert safe_spinner("dots") == "dots"


def test_safe_spinner_falls_back_on_cp1252():
    with mock.patch("sys.stdout", _stdout_with_encoding("cp1252")):
        assert safe_spinner("dots") == "line"


def test_safe_spinner_falls_back_on_ascii():
    with mock.patch("sys.stdout", _stdout_with_encoding("ascii")):
        assert safe_spinner("dots") == "line"


def test_safe_spinner_handles_missing_encoding():
    class _NoEncoding:
        encoding = None

    with mock.patch("sys.stdout", _NoEncoding()):
        # No encoding info → fall back conservatively to ASCII.
        assert safe_spinner("dots") == "line"
