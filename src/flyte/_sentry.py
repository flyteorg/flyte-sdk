"""
Sentry integration for Flyte SDK crash reporting.

Initializes Sentry with a hardcoded DSN to report errors from CLI commands
(e.g., `flyte start demo`). Users can opt out by setting FLYTE_DISABLE_SENTRY=true.
"""

import logging
import os

from flyte._logging import logger

_SENTRY_DSN = "https://d0e3f0a470b8e1333411eff583cf4004@o4507249423810560.ingest.us.sentry.io/4511135180128256"

_state = {"initialized": False}


def _is_dev_mode() -> bool:
    """Skip Sentry in dev mode (git checkout or dev version of flyte-sdk)."""
    from pathlib import Path

    if (Path(__file__).parent.parent.parent.parent / ".git").is_dir():
        return True

    version = _get_version()
    if version and "dev" in version.lower():
        return True

    return False


def _is_disabled() -> bool:
    return os.environ.get("FLYTE_DISABLE_SENTRY", "").lower() in ("true", "1", "yes")


def init() -> None:
    """Initialize Sentry SDK. Safe to call multiple times — only runs once."""
    if _state["initialized"]:
        return
    _state["initialized"] = True

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


def _iter_cause_chain(exc: BaseException):
    """Walk __cause__ / __context__ chain so wrapping doesn't hide the real type.

    Bounded depth - exception chains in the wild stay shallow (3-5 deep), but a
    bug elsewhere could create a cycle and we don't want this to spin.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    depth = 0
    while cur is not None and id(cur) not in seen and depth < 16:
        yield cur
        seen.add(id(cur))
        nxt = cur.__cause__ or cur.__context__
        cur = nxt
        depth += 1


def _is_user_error(exc: BaseException) -> bool:
    """Errors raised intentionally as user-facing messages — not crash reports."""
    try:
        import click

        click_user_exc: tuple[type, ...] = (click.Abort, click.exceptions.Exit, click.ClickException)
    except ImportError:
        click_user_exc = ()

    try:
        from flyte.errors import DeploymentError, ImageBuildError, InitializationError

        flyte_user_exc: tuple[type, ...] = (DeploymentError, ImageBuildError, InitializationError)
    except ImportError:
        flyte_user_exc = ()

    # Auth failures (expired refresh token, expired device code, IDP rejection)
    # are wrapped in RuntimeError("SelectCluster failed...") -> RuntimeSystemError
    # in _upload_single_file, so isinstance() on the outer exc misses them.
    # Walk __cause__ / __context__ to catch the original.
    try:
        from flyte.remote._client.auth.errors import AccessTokenNotFoundError, AuthenticationError

        auth_user_exc: tuple[type, ...] = (AccessTokenNotFoundError, AuthenticationError)
    except ImportError:
        auth_user_exc = ()

    user_excs = click_user_exc + flyte_user_exc + auth_user_exc
    if not user_excs:
        return False

    for cause in _iter_cause_chain(exc):
        if isinstance(cause, user_excs):
            return True
    return False


def capture_exception(exc: BaseException) -> None:
    """Capture an exception and send it to Sentry."""
    if _is_user_error(exc):
        return
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


def capture_errors(func):
    """Decorator that captures exceptions to Sentry and re-raises them."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            capture_exception(e)
            raise

    return wrapper


def count(key: str, value: int = 1, **tags: str) -> None:
    """Emit a counter metric to Sentry."""
    try:
        init()
        import sentry_sdk

        if sentry_sdk.is_initialized():
            sentry_sdk.metrics.count(key, value, attributes=tags or None)
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
