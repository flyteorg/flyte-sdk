"""Tests for flyte.app._types."""

import hashlib
import pathlib

import pytest

from flyte._image import Image
from flyte.app import AppEnvironment
from flyte.app._types import Domain, Subdomain
from flyte.models import SerializationContext


def _ctx(project="my-project", domain="my-domain") -> SerializationContext:
    return SerializationContext(
        org="my-org",
        project=project,
        domain=domain,
        version="v1",
        root_dir=pathlib.Path.cwd(),
    )


def _app_env(name="my-app") -> AppEnvironment:
    return AppEnvironment(name=name, image=Image.from_base("python:3.11"))


def test_subdomain_from_app_name_hash():
    subdomain = Subdomain.from_app_name("my-app")
    expected_hash = hashlib.sha256(b"my-project-my-domain").hexdigest()[:8]
    assert subdomain.resolve(_app_env(), _ctx()) == f"my-app-{expected_hash}"


def test_subdomain_hash_is_stable_per_project_domain():
    ctx = _ctx()
    resolved_a = Subdomain.from_app_name("app-a").resolve(_app_env("app-a"), ctx)
    resolved_b = Subdomain.from_app_name("app-b").resolve(_app_env("app-b"), ctx)
    # Same project/domain -> same hash suffix
    assert resolved_a.split("app-a-")[1] == resolved_b.split("app-b-")[1]
    # Different project -> different hash suffix
    other = Subdomain.from_app_name("app-a").resolve(_app_env("app-a"), _ctx(project="other"))
    assert other != resolved_a


def test_subdomain_from_app_name_default():
    subdomain = Subdomain.from_app_name("my-app", project_domain_suffix="default")
    assert subdomain.resolve(_app_env(), _ctx()) == "my-app-my-project-my-domain"


def test_subdomain_from_function():
    subdomain = Subdomain.from_function(lambda app_env, ctx: f"{app_env.name}.{ctx.org}.{ctx.domain}")
    assert subdomain.resolve(_app_env(), _ctx()) == "my-app.my-org.my-domain"


def test_subdomain_from_function_invalid_return():
    subdomain = Subdomain.from_function(lambda app_env, ctx: None)
    with pytest.raises(ValueError, match="non-empty str"):
        subdomain.resolve(_app_env(), _ctx())


def test_subdomain_invalid_suffix():
    with pytest.raises(ValueError, match="project_domain_suffix"):
        Subdomain.from_app_name("my-app", project_domain_suffix="bogus")


def test_subdomain_requires_app_name_or_function():
    with pytest.raises(ValueError, match="exactly one"):
        Subdomain()
    with pytest.raises(ValueError, match="exactly one"):
        Subdomain(app_name="my-app", function=lambda app_env, ctx: "x")


def test_subdomain_resolve_requires_project_and_domain():
    subdomain = Subdomain.from_app_name("my-app")
    with pytest.raises(ValueError, match="project and domain are required"):
        subdomain.resolve(_app_env(), _ctx(project=None))


def test_domain_accepts_subdomain():
    domain = Domain(subdomain=Subdomain.from_app_name("my-app"))
    assert isinstance(domain.subdomain, Subdomain)
    domain = Domain(subdomain="literal-subdomain")
    assert domain.subdomain == "literal-subdomain"
