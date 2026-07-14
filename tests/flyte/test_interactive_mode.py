"""Tests that with_runcontext(interactive_mode=...) can force interactive mode on or off."""

import pytest

import flyte._tools
from flyte._run import _Runner


@pytest.mark.parametrize(
    ("in_ipython", "interactive_mode", "expected"),
    [
        (True, None, True),     # Auto-detect notebook -> pkl bundle
        (False, None, False),   # Auto-detect script -> tgz bundle
        (True, False, False),   # Forced off wins over notebook detection
        (False, True, True),    # Forced on wins over script detection
    ],
)
def test_interactive_mode_override(monkeypatch, in_ipython, interactive_mode, expected):
    monkeypatch.setattr(flyte._tools, "ipython_check", lambda: in_ipython)
    runner = _Runner(interactive_mode=interactive_mode)
    assert runner._interactive_mode is expected
