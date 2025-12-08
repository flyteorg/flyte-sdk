from flyte.remote._console import _get_http_domain


def test_get_http_domain():
    # local gets overridden to 8080
    assert _get_http_domain("dns:///localhost:8090", True) == "http://localhost:8080"
    assert _get_http_domain("http://localhost", True) == "http://localhost:8080"
    assert _get_http_domain("dns:///example.com", False) == "https://example.com"
    assert _get_http_domain("https://example.com", False) == "https://example.com"
