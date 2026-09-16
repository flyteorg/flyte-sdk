class AccessTokenNotFoundError(RuntimeError):
    """
    This error is raised with Access token is not found or if Refreshing the token fails
    """


class AuthenticationError(RuntimeError):
    """
    This is raised for any AuthenticationError
    """


class AuthenticationTransientError(AuthenticationError):
    """
    This is raised when the token endpoint fails in a way that is worth retrying
    -- a 5xx or 429 from a proxy or gateway in front of the IDP, which carries no
    OAuth error code and often no JSON body at all.

    Subclasses AuthenticationError so callers outside a polling loop, which have
    nothing to retry with, keep treating it as the failure it is for them. The
    device-flow poll catches it specifically and keeps polling, because RFC 8628
    expects the client to keep asking until the code expires.
    """


class AuthenticationPending(RuntimeError):
    """
    This is raised if the token endpoint returns authentication pending
    """


class AuthenticationSlowDown(AuthenticationPending):
    """
    This is raised if the token endpoint returns slow_down, meaning the client
    is polling too fast and must widen its polling interval (RFC 8628 3.5).

    Subclasses AuthenticationPending so that callers which only care that the
    request is still outstanding keep working unchanged.
    """
