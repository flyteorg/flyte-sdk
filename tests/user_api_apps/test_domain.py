from flyte.app import Domain


def test_domain_defaults():
    d = Domain()
    assert d.subdomain is None
    assert d.custom_domain is None


def test_domain_subdomain():
    d = Domain(subdomain="my-app")
    assert d.subdomain == "my-app"
    assert d.custom_domain is None


def test_domain_custom_domain():
    d = Domain(custom_domain="app.example.com")
    assert d.custom_domain == "app.example.com"
    assert d.subdomain is None


def test_domain_both():
    d = Domain(subdomain="my-app", custom_domain="app.example.com")
    assert d.subdomain == "my-app"
    assert d.custom_domain == "app.example.com"
