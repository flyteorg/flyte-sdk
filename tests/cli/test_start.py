from flyte.cli._start import start


def test_start_includes_remote_tui_command():
    assert "remote-tui" in start.commands
