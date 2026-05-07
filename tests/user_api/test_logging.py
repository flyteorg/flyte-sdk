import json
import logging

import flyte
from flyte._logging import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_USER_LOG_LEVEL,
    JSONFormatter,
    get_env_log_level,
    get_env_user_log_level,
    is_rich_logging_disabled,
    log_format_from_env,
    make_hyperlink,
)


def test_logger_exists():
    assert flyte.logger is not None
    assert flyte.logger.name == "flyte.user"
    assert isinstance(flyte.logger, logging.Logger)


def test_default_log_level():
    assert DEFAULT_LOG_LEVEL == logging.WARNING


def test_get_env_log_level_default(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert get_env_log_level() == logging.WARNING


def test_get_env_log_level_named(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert get_env_log_level() == logging.DEBUG


def test_get_env_log_level_numeric(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "10")
    assert get_env_log_level() == 10


def test_get_env_log_level_invalid(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "invalid_level")
    assert get_env_log_level() == DEFAULT_LOG_LEVEL


def test_log_format_from_env_default(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    assert log_format_from_env() == "console"


def test_log_format_from_env_json(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    assert log_format_from_env() == "json"


def test_log_format_from_env_invalid(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "invalid")
    assert log_format_from_env() == "console"


def test_is_rich_logging_disabled_default(monkeypatch):
    monkeypatch.delenv("DISABLE_RICH_LOGGING", raising=False)
    assert is_rich_logging_disabled() is False


def test_is_rich_logging_disabled_set(monkeypatch):
    monkeypatch.setenv("DISABLE_RICH_LOGGING", "1")
    assert is_rich_logging_disabled() is True


def test_make_hyperlink():
    result = make_hyperlink("Click here", "https://example.com")
    assert "Click here" in result
    assert "https://example.com" in result


def test_json_formatter():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=None,
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "Test message"
    assert parsed["level"] == "INFO"
    assert "timestamp" in parsed


def test_user_logger_exists():
    assert flyte.logger is not None
    assert flyte.logger.name == "flyte.user"
    assert isinstance(flyte.logger, logging.Logger)


def test_user_logger_default_level():
    assert DEFAULT_USER_LOG_LEVEL == logging.INFO


def test_user_logger_independent_of_internal_level():
    from flyte._logging import logger as internal_logger

    original_internal = internal_logger.level
    original_user = flyte.logger.level
    try:
        internal_logger.setLevel(logging.CRITICAL)
        assert flyte.logger.level != logging.CRITICAL
    finally:
        internal_logger.setLevel(original_internal)
        flyte.logger.setLevel(original_user)


def test_user_log_level_env_var(monkeypatch):
    monkeypatch.setenv("USER_LOG_LEVEL", "debug")
    assert get_env_user_log_level() == logging.DEBUG


def test_user_log_level_env_var_default(monkeypatch):
    monkeypatch.delenv("USER_LOG_LEVEL", raising=False)
    assert get_env_user_log_level() == logging.INFO


def test_user_logger_no_flyte_prefix():
    """The user logger's formatter must not stamp the [flyte] internal prefix."""
    from flyte._logging import ContextFormatter

    for handler in flyte.logger.handlers:
        formatter = handler.formatter
        if isinstance(formatter, ContextFormatter):
            assert not formatter._internal_prefix, "user_logger formatter must not use internal_prefix"


def test_user_logger_no_flyte_prefix_after_rich_init():
    """
    Regression: when initialize_logger(enable_rich=True) is called (e.g. via flyte.init()),
    the rich handler attached to the user logger must not carry an internal_prefix formatter.
    """
    from flyte._logging import ContextFormatter, initialize_logger

    initialize_logger(enable_rich=True)
    try:
        for handler in flyte.logger.handlers:
            formatter = handler.formatter
            if isinstance(formatter, ContextFormatter):
                assert not formatter._internal_prefix, (
                    "user_logger formatter must not use internal_prefix even with rich handler"
                )
    finally:
        initialize_logger(enable_rich=False)


def test_json_formatter_with_context():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test",
        args=None,
        exc_info=None,
    )
    record.run_name = "my-run"
    record.action_name = "my-action"
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["run_name"] == "my-run"
    assert parsed["action_name"] == "my-action"
