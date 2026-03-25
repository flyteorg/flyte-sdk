def hostname_from_url(url: str) -> str:
    """Parse a URL and return the hostname part."""

    # Handle dns:/// format specifically (gRPC convention)
    if url.startswith("dns:///"):
        return url[7:]  # Skip the "dns:///" prefix

    # Handle standard URL formats
    import urllib.parse

    # If no scheme, prepend one so urlparse doesn't misinterpret host:port
    if "://" not in url:
        parsed = urllib.parse.urlparse(f"dummy://{url}")
    else:
        parsed = urllib.parse.urlparse(url)
    return parsed.netloc or parsed.path.lstrip("/").rsplit("/")[0]


def org_from_endpoint(endpoint: str | None) -> str | None:
    """
    Extracts the organization from the endpoint URL. The organization is assumed to be the first part of the domain.
    This is temporary until we have a proper organization discovery mechanism through APIs.

    :param endpoint: The endpoint URL
    :return: The organization name or None if not found
    """
    if not endpoint:
        return None

    hostname = hostname_from_url(endpoint)
    # Strip port if present
    hostname = hostname.split(":")[0]
    domain_parts = hostname.split(".")
    if len(domain_parts) >= 1:
        # Return the first part of the domain (org for multi-part, or hostname itself for localhost)
        return domain_parts[0]
    return None


def sanitize_endpoint(endpoint: str | None) -> str | None:
    """
    Sanitize the endpoint URL by ensuring it has a valid scheme.
    :param endpoint: The endpoint URL to sanitize
    :return: Sanitized endpoint URL or None if the input was None
    """
    if not endpoint:
        return None
    if "://" not in endpoint:
        endpoint = f"dns:///{endpoint}"
    else:
        if endpoint.startswith("https://"):
            # If the endpoint starts with dns:///, we assume it's a gRPC endpoint
            endpoint = f"dns:///{endpoint[8:]}"
        elif endpoint.startswith("http://"):
            # If the endpoint starts with http://, we assume it's a REST endpoint
            endpoint = f"dns:///{endpoint[7:]}"
        elif not endpoint.startswith("dns:///"):
            raise RuntimeError(
                f"Invalid endpoint {endpoint}, expected format is "
                f"dns:///<hostname> or https://<hostname> or http://<hostname>"
            )
    endpoint = endpoint.removesuffix("/")
    return endpoint
