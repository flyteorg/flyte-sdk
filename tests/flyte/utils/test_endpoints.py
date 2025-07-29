import pytest

from flyte._utils import org_discovery


@pytest.mark.parametrize(
    "url,expected",
    [
        ("dns:///foo.bar.com", "foo.bar.com"),
        ("https://foo.bar.com/path", "foo.bar.com"),
        ("http://foo.bar.com:8080/path", "foo.bar.com:8080"),
        ("ftp://foo.bar.com", "foo.bar.com"),
        ("foo.bar.com", "foo.bar.com"),
        ("dns:///foo.bar.com:1234", "foo.bar.com:1234"),
        ("https://foo.bar.com", "foo.bar.com"),
        ("http://localhost:8000", "localhost:8000"),
        ("dns:///localhost", "localhost"),
        ("", ""),
    ],
)
def test_hostname_from_url(url, expected):
    assert org_discovery.hostname_from_url(url) == expected


@pytest.mark.parametrize(
    "endpoint,expected_org",
    [
        ("https://foo.bar.com/path", "foo"),
        ("dns:///foo.bar.com", "foo"),
        ("http://foo.bar.com:8080/path", "foo"),
        ("ftp://foo.bar.com", "foo"),
        ("foo.bar.com", "foo"),
        ("dns:///foo.bar.com:1234", "foo"),
        ("https://foo.bar.com", "foo"),
        ("http://localhost:8000", None),
        ("dns:///localhost", None),
        ("", None),
        (None, None),
        ("https://bar.com", None),
        ("bar.com", None),
        ("dns:///bar.com", None),
        ("https://foo.bar.baz.com", "foo"),
        ("dns:///foo.bar.baz.com", "foo"),
    ],
)
def test_org_from_endpoint(endpoint, expected_org):
    assert org_discovery.org_from_endpoint(endpoint) == expected_org


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("https://foo.bar.com/path", "dns:///foo.bar.com/path"),
        ("dns:///foo.bar.com", "dns:///foo.bar.com"),
        ("http://foo.bar.com:8080/path", "dns:///foo.bar.com:8080/path"),
        ("foo.bar.com", "dns:///foo.bar.com"),
        ("dns:///foo.bar.com:1234", "dns:///foo.bar.com:1234"),
        ("https://foo.bar.com", "dns:///foo.bar.com"),
        ("http://localhost:8000", "dns:///localhost:8000"),
        ("dns:///localhost", "dns:///localhost"),
        ("", None),
        (None, None),
    ],
)
def test_sanitize_endpoint(endpoint, expected):
    assert org_discovery.sanitize_endpoint(endpoint) == expected


def test_sanitize_endpoint_invalid():
    with pytest.raises(RuntimeError, match="Invalid endpoint"):
        org_discovery.sanitize_endpoint("invalid://foo.bar.com")
    with pytest.raises(RuntimeError, match="Invalid endpoint"):
        org_discovery.sanitize_endpoint("ftp://foo.bar.com:8080/path")  # Should be dns:/// format
