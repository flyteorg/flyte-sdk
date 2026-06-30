import io
import logging
import pathlib

from rich.console import Console
from rich.logging import RichHandler

from flyte._logging import logger
from flyte.config._reader import read_file_if_exists


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
