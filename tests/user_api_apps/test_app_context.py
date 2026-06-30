from flyte.app import ctx
from flyte.app._context import AppContext


def test_ctx_returns_app_context(monkeypatch):
    monkeypatch.delenv("_RUN_MODE", raising=False)
    monkeypatch.delenv("FLYTE_INTERNAL_EXECUTION_PROJECT", raising=False)
    monkeypatch.delenv("FLYTE_INTERNAL_EXECUTION_DOMAIN", raising=False)
    result = ctx()
    assert isinstance(result, AppContext)
    assert result.mode == "remote"
    assert result.project == ""
    assert result.domain == ""


def test_ctx_reads_env_vars(monkeypatch):
    monkeypatch.setenv("_RUN_MODE", "local")
    monkeypatch.setenv("FLYTE_INTERNAL_EXECUTION_PROJECT", "my-project")
    monkeypatch.setenv("FLYTE_INTERNAL_EXECUTION_DOMAIN", "development")
    result = ctx()
    assert result.mode == "local"
    assert result.project == "my-project"
    assert result.domain == "development"


def test_app_context_frozen():
    ac = AppContext(mode="local", project="proj", domain="dev")
    assert ac.mode == "local"
    assert ac.project == "proj"
    assert ac.domain == "dev"
