"""Tell apart why a ConnectRPC call came back UNAVAILABLE.

connectrpc's client wraps everything it does to send a request (encoding the message,
compressing it, handing it to the HTTP transport, validating the response) in one
`except Exception` that re-raises as `ConnectError(Code.UNAVAILABLE, str(exc))` chained to the
original exception. A refused connection or a reset stream lands there, but so does a bug or an
incompatibility in this process, for example a library that misbehaves on a new Python version.
Reporting the latter as "the service is unavailable" sends people to check a service that never
saw the request.

An UNAVAILABLE the server sent has no `__cause__`, and a network failure is chained to an
`OSError` (pyqwest raises `ConnectionError` for refused connections, DNS and TLS failures) or to
one of pyqwest's read, write or HTTP/2 stream errors. Anything else is a local failure.
"""

from __future__ import annotations

import platform

from connectrpc.code import Code
from connectrpc.errors import ConnectError


def _is_network_error(exc: BaseException) -> bool:
    if isinstance(exc, (OSError, TimeoutError)):
        return True
    try:
        from pyqwest import ReadError, StreamError, WriteError
    except ImportError:
        return False
    return isinstance(exc, (StreamError, ReadError, WriteError))


def local_failure(e: BaseException) -> BaseException | None:
    """Return the exception behind an UNAVAILABLE that this process raised itself, else None.

    None means the error came from the server or the network, so "unavailable" is accurate.
    """
    if not isinstance(e, ConnectError) or e.code != Code.UNAVAILABLE:
        return None
    cause = e.__cause__
    if cause is None or _is_network_error(cause):
        return None
    return cause


def describe_local_failure(cause: BaseException, action: str) -> str:
    """A message for a request that failed in this process before it reached the service."""
    try:
        from flyte._version import __version__
    except ImportError:
        __version__ = "unknown"

    return (
        f"{action} failed in this process before the request reached the Flyte service: "
        f"{type(cause).__name__}: {cause}. "
        f"Python {platform.python_version()}, flyte {__version__}. "
        "This is a client-side error, not a service outage."
    )
