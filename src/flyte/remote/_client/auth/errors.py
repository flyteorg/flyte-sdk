class AccessTokenNotFoundError(RuntimeError):
    """
    This error is raised with Access token is not found or if Refreshing the token fails
    """


class AuthenticationError(RuntimeError):
    """
    This is raised for any AuthenticationError
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
