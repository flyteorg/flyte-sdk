"""
FastAPI middleware for automatic Flyte authentication passthrough.

This module provides middleware that automatically extracts authentication headers
from incoming requests and sets them in the Flyte context, eliminating the need
for manual auth_metadata() wrapping in every endpoint.

Example:
    Basic usage with default extractors (Authorization + Cookie headers)::

        from fastapi import FastAPI
        from flyte.app.extras import FastAPIAuthMiddleware

        app = FastAPI()
        app.add_middleware(FastAPIAuthMiddleware, excluded_paths={"/health"})

        @app.get("/me")
        async def get_current_user():
            # Auth metadata automatically set from request headers
            user = await remote.User.get.aio()
            return {"subject": user.subject()}

    Advanced usage with custom extractors and path exclusions::

        from flyte.app.extras import (
            FastAPIAuthMiddleware,
            extract_authorization_header,
            extract_custom_header,
        )

        app.add_middleware(
            FastAPIAuthMiddleware,
            header_extractors=[
                extract_authorization_header,
                extract_custom_header("x-api-key"),
            ],
            excluded_paths={"/health", "/metrics"},
        )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)

# Header extractor type: takes a Request, returns (key, value) tuple or None
HeaderExtractor = Callable[["Request"], tuple[str, str] | None]


def extract_authorization_header(request: "Request") -> tuple[str, str] | None:
    """
    Extract the Authorization header from the request.

    Args:
        request: The FastAPI/Starlette request object

    Returns:
        Tuple of ("authorization", header_value) if present, None otherwise
    """
    auth_header = request.headers.get("authorization")
    if auth_header:
        return ("authorization", auth_header)
    return None


def extract_cookie_header(request: "Request") -> tuple[str, str] | None:
    """
    Extract the Cookie header from the request.

    Args:
        request: The FastAPI/Starlette request object

    Returns:
        Tuple of ("cookie", header_value) if present, None otherwise
    """
    cookie_header = request.headers.get("cookie")
    if cookie_header:
        return ("cookie", cookie_header)
    return None


def extract_custom_header(header_name: str) -> HeaderExtractor:
    """
    Create a header extractor for a custom header name.

    Args:
        header_name: The name of the header to extract (case-insensitive)

    Returns:
        A header extractor function that extracts the specified header

    Example::

        # Create extractor for X-API-Key header
        api_key_extractor = extract_custom_header("x-api-key")

        app.add_middleware(
            FastAPIAuthMiddleware,
            header_extractors=[api_key_extractor],
        )
    """

    def extractor(request: "Request") -> tuple[str, str] | None:
        header_value = request.headers.get(header_name.lower())
        if header_value:
            return (header_name.lower(), header_value)
        return None

    extractor.__name__ = f"extract_{header_name.replace('-', '_')}_header"
    return extractor


class FastAPIAuthMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that automatically sets Flyte auth metadata from request headers.

    This middleware extracts authentication headers from incoming HTTP requests and
    sets them in the Flyte context using the auth_metadata() context manager. This
    eliminates the need to manually wrap endpoint handlers with auth_metadata().

    The middleware is highly configurable:
    - Custom header extractors can be provided
    - Specific paths can be excluded from auth requirements
    - Auth can be optional or required

    Attributes:
        header_extractors: List of functions to extract headers from requests
        excluded_paths: Set of URL paths that bypass auth extraction
        require_auth: If True, return 401 when no auth headers found

    Thread Safety:
        This middleware is async-safe and properly isolates auth metadata per request
        using Python's contextvars. Multiple concurrent requests with different
        authentication will not interfere with each other.
    """

    def __init__(
        self,
        app,
        header_extractors: list[HeaderExtractor] | None = None,
        excluded_paths: set[str] | None = None,
    ):
        """
        Initialize the Flyte authentication middleware.

        Args:
            app: The FastAPI/Starlette application
            header_extractors: List of functions to extract headers. Each function
                takes a Request and returns (key, value) tuple or None.
                Defaults to [extract_authorization_header, extract_cookie_header].
            excluded_paths: Set of URL paths to exclude from auth extraction.
                Requests to these paths proceed without setting auth context.
        """
        super().__init__(app)

        if header_extractors is None:
            self.header_extractors = [
                extract_authorization_header,
                extract_cookie_header,
            ]
        else:
            self.header_extractors = header_extractors

        self.excluded_paths = excluded_paths or set()

    async def dispatch(self, request: "Request", call_next) -> "Response":
        """
        Process each request, extracting auth headers and setting Flyte context.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler to call

        Returns:
            The HTTP response from the handler
        """

        # Skip auth extraction for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Extract auth headers using all configured extractors
        auth_tuples = []
        for extractor in self.header_extractors:
            try:
                result = extractor(request)
                if result is not None:
                    auth_tuples.append(result)
            except Exception as e:
                logger.warning(f"Header extractor {extractor.__name__} failed: {e}")

        # Require auth headers
        if not auth_tuples:
            logger.info("No auth tuples found")
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication credentials required"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Set auth metadata in Flyte context for the duration of this request
        from flyte.remote._auth_metadata import auth_metadata

        with auth_metadata(*auth_tuples):
            response = await call_next(request)

        return response
