"""
Sentry integration for Flyte SDK crash reporting.

Initializes Sentry with a hardcoded DSN to report errors from CLI commands
(e.g., `flyte start demo`). Users can opt out by setting FLYTE_DISABLE_SENTRY=true.
"""

import atexit
import errno
import logging
import os
from contextlib import contextmanager

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
        # count() doesn't flush per call, flush everything once at exit.
        atexit.register(lambda: sentry_sdk.flush(timeout=2))
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


_USER_ACTIONABLE_CONNECT_CODES: frozenset[str] = frozenset(
    {
        # User/config problems — backend rejects the request as invalid.
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
        "FAILED_PRECONDITION",
        "INVALID_ARGUMENT",
        "NOT_FOUND",
        "ALREADY_EXISTS",
        # Endpoint/version skew — the control-plane route the SDK called isn't
        # served (HTTP 404 → connect maps to UNIMPLEMENTED, not NOT_FOUND). This
        # is a wrong endpoint / incompatible backend version / misrouted ingress,
        # never a Python-SDK logic bug, so the SDK can't recover from it.
        "UNIMPLEMENTED",
        # Transient infra / availability problems — DNS lookup failed, TCP
        # connect refused, connection reset, request timed out. The SDK
        # cannot recover from these, so they shouldn't be crash-reported.
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
    }
)


def _is_user_actionable_connect_error(exc: BaseException) -> bool:
    """ConnectError responses the SDK cannot recover from.

    Two flavors live in the same filter set:

    * User/config problems (UNAUTHENTICATED, PERMISSION_DENIED, INVALID_ARGUMENT, …)
      — the backend rejects the request as invalid; the CLI's InvokeBaseMixin
      (flyte/cli/_common.py) already maps these to ClickException.
    * Transient infrastructure problems (UNAVAILABLE, DEADLINE_EXCEEDED) — DNS
      lookup failures, TCP connect refused, connection reset, request timed out
      against the cluster service. These are not SDK bugs; INTERNAL is
      intentionally NOT filtered because it can indicate a real bug.
    * Endpoint/version skew (UNIMPLEMENTED) — a raw HTTP 404 from the ingress or
      a missing control-plane route (connect maps 404 → UNIMPLEMENTED). The
      endpoint config is wrong or the backend doesn't serve that RPC; the SDK
      can't recover (FLYTE-SDK-4F: SelectCluster "Not Found" → RuntimeSystemError).

    Code paths outside the CLI (capture_exception in _run.py, capture_errors on
    deploy) surface RuntimeSystemError wrappers whose cause chain still
    terminates in a ConnectError, and those leak into Sentry as if they were
    SDK bugs. The cause-chain walk in _is_user_error catches them here.
    """
    try:
        from connectrpc.errors import ConnectError
    except ImportError:
        return False
    if not isinstance(exc, ConnectError):
        return False
    code = getattr(exc, "code", None)
    return getattr(code, "name", None) in _USER_ACTIONABLE_CONNECT_CODES


_USER_ENVIRONMENT_OSERROR_ERRNOS: frozenset[int] = frozenset({errno.ENOSPC})


def _is_user_environment_oserror(exc: BaseException) -> bool:
    """OSError variants caused by the user's local environment, not SDK bugs.

    ENOSPC ("No space left on device") surfaces from shutil._fastcopy_sendfile
    during `flyte deploy` bundle uploads when the user's machine is out of disk
    (FLYTE-SDK-32). Disk-full is a user environment problem, not something the
    SDK can fix, so it shouldn't be reported as a crash.
    """
    if not isinstance(exc, OSError):
        return False
    return exc.errno in _USER_ENVIRONMENT_OSERROR_ERRNOS


def _is_transient_network_error(exc: BaseException) -> bool:
    """Transient network / connectivity failures, not SDK bugs.

    The `flyte deploy`/`flyte run` upload path (flyte.remote._data) reaches the
    cluster service (SelectCluster) and then PUTs the bundle to a signed object
    store URL. Both legs ride the user's network, so flaky links, VPN drops,
    refused/reset connections and request timeouts surface here as
    RuntimeSystemError wrappers whose cause chain terminates in a timeout or a
    transport-level connection error. None of those are something the SDK can
    fix, so they shouldn't be reported as crashes. Covers, among others:

    - FLYTE-SDK-29: SelectCluster ``TimeoutError`` ("Request timed out")
    - FLYTE-SDK-47: builtin ``ConnectionError`` ("Connection refused")
    - FLYTE-SDK-3W: ``httpx.WriteError`` ("Connection reset by peer")
    - FLYTE-SDK-36: ``httpx.ReadError`` during the signed-URL upload
    - FLYTE-SDK-4M: ``httpx.RemoteProtocolError`` ("Server disconnected without
      sending a response") — the object store dropped the PUT mid-flight

    Transient ConnectError status codes (DEADLINE_EXCEEDED / UNAVAILABLE) are
    handled by ``_is_user_actionable_connect_error`` and intentionally not
    duplicated here. INTERNAL / UNKNOWN stay reported — they can signal a real
    backend bug worth tracking.
    """
    # Builtin timeouts (asyncio.TimeoutError and socket.timeout are both aliases
    # of TimeoutError on 3.11+) and connection errors (ConnectionRefused/Reset/
    # Aborted/BrokenPipe) are all OSError subclasses raised by the network stack,
    # never by SDK logic.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    # httpx transport failures from the signed-URL PUT. TimeoutException and
    # NetworkError are the two transport-error families (ConnectTimeout,
    # ReadTimeout, ReadError, WriteError, ConnectError, PoolTimeout, ...).
    # RemoteProtocolError (a ProtocolError, not a NetworkError) is the server
    # hanging up mid-response. None are OSError subclasses, so check explicitly.
    try:
        import httpx

        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
            return True
    except ImportError:
        pass

    return False


def _is_user_error(exc: BaseException) -> bool:
    """Errors raised intentionally as user-facing messages — not crash reports."""
    try:
        import click

        click_user_exc: tuple[type, ...] = (click.Abort, click.exceptions.Exit, click.ClickException)
    except ImportError:
        click_user_exc = ()

    try:
        from flyte.errors import InitializationError, RuntimeUserError

        # RuntimeUserError is the parent class of ModuleLoadError, DeploymentError,
        # ImageBuildError, OOMError, TaskTimeoutError, RuntimeDataValidationError,
        # CodeBundleError, etc. — all "this is your code/config, not an SDK bug"
        # errors. InitializationError is a sibling BaseRuntimeError, also user-facing.
        flyte_user_exc: tuple[type, ...] = (RuntimeUserError, InitializationError)
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

    for cause in _iter_cause_chain(exc):
        if user_excs and isinstance(cause, user_excs):
            return True
        if _is_user_actionable_connect_error(cause):
            return True
        if _is_user_environment_oserror(cause):
            return True
        if _is_transient_network_error(cause):
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
    except ImportError:
        pass
    except Exception:
        pass


@contextmanager
def track_operation(operation: str):
    """Count success/failure of a key SDK operation."""
    try:
        yield
    except BaseException as e:
        tags = {
            "operation": operation,
            "status": "error",
            "error_type": type(e).__name__,
            "error_kind": "user" if _is_user_error(e) else "system",
        }
        # .code is a stable, low-cardinality failure mode (unlike str(e)).
        code = getattr(e, "code", None)
        if code is not None:
            # ConnectError.code is an enum (use .name); flyte BaseRuntimeError.code
            # is a str (no .name, fall back to str()).
            tags["error_code"] = getattr(code, "name", None) or str(code)
        count("flyte.operation", **tags)
        raise
    else:
        count("flyte.operation", operation=operation, status="success")


def _get_version() -> str | None:
    try:
        from flyte._version import __version__

        return __version__
    except Exception:
        return None
