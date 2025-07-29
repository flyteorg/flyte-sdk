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
