import pytest

from flyte.app import AppEndpoint


def test_app_endpoint_basic():
    ae = AppEndpoint(app_name="upstream-app")
    assert ae.app_name == "upstream-app"
    assert ae.public is False
    assert ae.type == "string"


def test_app_endpoint_public():
    ae = AppEndpoint(app_name="upstream-app", public=True)
    assert ae.public is True


def test_app_endpoint_type_always_string():
    ae = AppEndpoint(app_name="my-app")
    assert ae.type == "string"


@pytest.mark.asyncio
async def test_app_endpoint_materialize_returns_self():
    ae = AppEndpoint(app_name="upstream-app")
    result = await ae.materialize()
    assert result is ae


def test_app_endpoint_json_roundtrip():
    ae = AppEndpoint(app_name="upstream-app", public=True)
    json_str = ae.model_dump_json()
    restored = AppEndpoint.model_validate_json(json_str)
    assert restored.app_name == "upstream-app"
    assert restored.public is True
    assert restored.type == "string"
