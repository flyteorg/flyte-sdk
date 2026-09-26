import io
import logging
import pathlib

import pytest
from rich.console import Console
from rich.logging import RichHandler

from flyte._logging import logger
from flyte.config._reader import ConfigEntry, YamlConfigEntry, read_file_if_exists
from flyte.errors import InitializationError


def _boom(_value):
    """Stand-in for a transform whose parser raises without naming its source."""
    raise ValueError("No closing quotation")


def test_read_file_if_exists_debug_log_renders_under_rich_markup(tmp_path: pathlib.Path) -> None:
    """Regression: the debug log embeds the file path inside square brackets. Under
    rich>=15 the markup parser strictly rejects `[/path]` as a closing tag with no
    matching opener, raising MarkupError and crashing the caller. The call site
    must escape the surrounding brackets so the message renders cleanly when a
    RichHandler with markup=True is installed (matches flyte._logging setup).
    """
    buf = io.StringIO()
    handler = RichHandler(
        console=Console(file=buf, force_terminal=False, width=400, color_system=None),
        markup=True,
        show_time=False,
        show_path=False,
        show_level=False,
    )
    prior_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        target = tmp_path / "secret"
        target.write_text("dummy")

        # Would raise rich.errors.MarkupError if the path's leading `/` made
        # the bracketed segment look like a closing tag.
        assert read_file_if_exists(str(target)) == "dummy"

        output = buf.getvalue()
        assert f"[{target}]" in output
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)


class TestEnvTransformFailure:
    """A malformed env var must name itself.

    An env var can only ever carry a string, so whatever parsing a setting needs
    happens in the entry's `transform`. When that parsing fails, the parser's own
    exception says nothing about where the value came from -- `shlex.split` on an
    unbalanced quote raises a bare `ValueError: No closing quotation`. The read
    must turn it into a classified error naming the variable and the setting.

    The file path deliberately keeps swallowing a bad value: this is only about a
    variable the caller explicitly set, which must not be silently ignored and
    then resurface later as an auth failure.
    """

    def test_transform_failure_names_the_env_var_and_the_setting(self, monkeypatch) -> None:
        entry = ConfigEntry(YamlConfigEntry("admin.command", list), transform=_boom)
        monkeypatch.setenv("FLYTE_ADMIN_COMMAND", "whatever")

        with pytest.raises(InitializationError) as exc:
            entry.read()

        message = str(exc.value)
        assert "FLYTE_ADMIN_COMMAND" in message
        assert "admin.command" in message
        assert "'whatever'" in message, "the rejected value belongs in the message"

    def test_alias_is_reported_when_the_alias_carried_the_value(self, monkeypatch) -> None:
        """The message must name the variable actually set, not the preferred one."""
        entry = ConfigEntry(YamlConfigEntry("admin.command", list, aliases=("FLYTE_AUTH_COMMAND",)), transform=_boom)
        monkeypatch.delenv("FLYTE_AUTH_COMMAND", raising=False)
        monkeypatch.setenv("FLYTE_ADMIN_COMMAND", "whatever")

        with pytest.raises(InitializationError, match="FLYTE_ADMIN_COMMAND"):
            entry.read()

    def test_already_classified_error_passes_through_unchanged(self, monkeypatch) -> None:
        """A transform that has already done the classifying keeps its own message."""
        sentinel = InitializationError("MineNotYours", "user", "a better message")

        def _raise_classified(_v):
            raise sentinel

        entry = ConfigEntry(YamlConfigEntry("admin.command", list), transform=_raise_classified)
        monkeypatch.setenv("FLYTE_ADMIN_COMMAND", "whatever")

        with pytest.raises(InitializationError) as exc:
            entry.read()

        assert exc.value is sentinel

    def test_a_good_value_is_untouched(self, monkeypatch) -> None:
        entry = ConfigEntry(YamlConfigEntry("admin.command", list), transform=lambda v: v.split())
        monkeypatch.setenv("FLYTE_ADMIN_COMMAND", "uctl get-token")

        assert entry.read() == ["uctl", "get-token"]
