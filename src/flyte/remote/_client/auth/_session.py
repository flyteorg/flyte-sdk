from urllib.parse import urlparse


def normalize_rpc_endpoint(endpoint: str, *, insecure: bool = False) -> str:
    """Translate gRPC-style endpoint to http(s) URL for ConnectRPC."""
    scheme = "http" if insecure else "https"
    parsed = urlparse(endpoint)

    if parsed.scheme in ("http", "https"):
        return endpoint

    if parsed.scheme == "dns":
        host = parsed.path.lstrip("/")
        return f"{scheme}://{host}"

    # urlparse("example.com:8089") mis-parses "example.com" as the scheme and
    # leaves netloc empty.  A genuine URL like "ftp://example.com" will have a
    # non-empty netloc.  Use that to tell the two cases apart.
    if parsed.netloc:
        # A real URL with an unrecognised scheme (e.g. ftp://).
        raise ValueError(
            f"Unknown scheme '{parsed.scheme}' in endpoint '{endpoint}'. "
            "Use http://, https://, dns:///, or bare host:port."
        )

    # Bare host:port (no scheme at all, or urlparse mis-detected one).
    return f"{scheme}://{endpoint}"
