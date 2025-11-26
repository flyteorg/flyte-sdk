from urllib.parse import urlparse


def _get_http_domain(endpoint: str, insecure: bool) -> str:
    scheme = "http" if insecure else "https"
    parsed = urlparse(endpoint)
    if parsed.scheme == "dns":
        domain = parsed.path.lstrip("/")
    else:
        domain = parsed.netloc or parsed.path
    # TODO: make console url configurable
    domain_split = domain.split(":")
    if domain_split[0] == "localhost":
        # Always use port 8080 for localhost, until the to do is done.
        domain = "localhost:8080"
    return f"{scheme}://{domain}"


def get_run_url(endpoint: str, insecure: bool, project: str, domain: str, run_name: str) -> str:
    return f"{_get_http_domain(endpoint, insecure)}/v2/runs/project/{project}/domain/{domain}/{run_name}"
