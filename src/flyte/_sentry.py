"""
Sentry integration for Flyte SDK crash reporting.

Initializes Sentry with a hardcoded DSN to report errors from CLI commands
(e.g., `flyte start demo`). Users can opt out by setting FLYTE_DISABLE_SENTRY=true.
"""

import logging
import os

logger = logging.getLogger(__name__)

_SENTRY_DSN = "https://d0e3f0a470b8e1333411eff583cf4004@o4507249423810560.ingest.us.sentry.io/4511135180128256"

_initialized = False


def _is_dev_mode() -> bool:
    """Skip Sentry in dev mode (running from a git checkout of flyte-sdk)."""
    from pathlib import Path

    return (Path(__file__).parent.parent.parent.parent / ".git").is_dir()


def _is_disabled() -> bool:
    return os.environ.get("FLYTE_DISABLE_SENTRY", "").lower() in ("true", "1", "yes")


def init() -> None:
    """Initialize Sentry SDK. Safe to call multiple times — only runs once."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    if _is_disabled() or _is_dev_mode():
        return

    try:
        import sentry_sdk

        # Silence Sentry's own error loggers (no noise if offline)
        logging.getLogger("sentry.errors").disabled = True
        logging.getLogger("sentry.errors.uncaught").disabled = True

        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            release=_get_version(),
            default_integrations=False,
        )
    except ImportError:
        pass
    except Exception:
        logger.debug("Failed to initialize Sentry", exc_info=True)


def capture_exception(exc: BaseException) -> None:
    """Capture an exception and send it to Sentry."""
    try:
        init()
        import sentry_sdk

        if sentry_sdk.is_initialized():
            sentry_sdk.capture_exception(exc)
            sentry_sdk.flush(timeout=2)
    except ImportError:
        pass
    except Exception:
        pass


def capture_message(msg: str) -> None:
    """Capture a message and send it to Sentry."""
    try:
        init()
        import sentry_sdk

        if sentry_sdk.is_initialized():
            sentry_sdk.capture_message(msg)
            sentry_sdk.flush(timeout=2)
    except ImportError:
        pass
    except Exception:
        pass


def count(key: str, value: int = 1, **tags: str) -> None:
    """Emit a counter metric to Sentry."""
    try:
        init()
        import sentry_sdk

        if sentry_sdk.is_initialized():
            sentry_sdk.metrics.count(key, value, attributes=tags if tags else None)
            sentry_sdk.flush(timeout=2)
    except ImportError:
        pass
    except Exception:
        pass


def _get_version() -> str | None:
    try:
        from flyte._version import __version__

        return __version__
    except Exception:
        return None
