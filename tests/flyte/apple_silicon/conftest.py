import pytest


@pytest.fixture(autouse=True)
def _isolated_apple_silicon_state(tmp_path_factory, monkeypatch):
    """Never touch the developer's real ~/.flyte/apple-silicon (secret, live bridge) from a test."""
    from flyte._internal.apple_silicon import _host

    monkeypatch.setattr(_host, "STATE_DIR", tmp_path_factory.mktemp("apple-silicon-state"))
