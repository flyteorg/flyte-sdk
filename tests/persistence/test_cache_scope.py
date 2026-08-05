import types
from pathlib import Path

import pytest

import flyte._initialize as _init
from flyte._initialize import _InitConfig
from flyte._persistence._db import _cache_scope
from flyte.errors import InitializationError


@pytest.fixture(autouse=True)
def _reset_init_config():
    """Save, clear, and restore the global init config around each test."""
    saved = _init._init_config
    _init._init_config = None
    yield
    _init._init_config = saved


def _set_init(*, project=None, domain=None, endpoint=None):
    """Install an init config singleton, with an optional client carrying an endpoint."""
    client = types.SimpleNamespace(endpoint=endpoint) if endpoint is not None else None
    _init._init_config = _InitConfig(
        root_dir=Path.cwd(),
        project=project,
        domain=domain,
        client=client,
    )


def test_cache_scope_uses_endpoint_project_domain_from_init_config():
    _set_init(project="proj", domain="dev", endpoint="dns:///example.union.ai")
    assert _cache_scope() == "dns:///example.union.ai:proj:dev"


def test_cache_scope_endpoint_empty_when_no_client():
    _set_init(project="proj", domain="dev", endpoint=None)
    assert _cache_scope() == ":proj:dev"


def test_cache_scope_empty_when_project_and_domain_unset():
    _set_init(project=None, domain=None, endpoint=None)
    assert _cache_scope() == "::"


def test_cache_scope_reflects_init_overrides_not_defaults():
    """The scope must track the project/domain the run was initialized with."""
    _set_init(project="flytesnacks", domain="development", endpoint="dns:///dogfood.union.ai")
    first = _cache_scope()

    # A different init (e.g. another domain) must yield a different scope so the
    # two environments never share cache entries.
    _set_init(project="flytesnacks", domain="staging", endpoint="dns:///dogfood.union.ai")
    second = _cache_scope()

    assert first != second
    assert first.endswith(":flytesnacks:development")
    assert second.endswith(":flytesnacks:staging")


def test_cache_scope_raises_when_not_initialized():
    # _init_config is None via the autouse fixture.
    with pytest.raises(InitializationError):
        _cache_scope()
