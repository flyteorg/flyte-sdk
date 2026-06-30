"""
Unit tests for init_from_api_key functionality
"""

import base64
import os
from unittest import mock

import pytest

from flyte import init_from_api_key
from flyte.errors import InitializationError


def create_encoded_api_key(endpoint: str, client_id: str, client_secret: str, org: str) -> str:
    """Helper function to create an encoded API key."""
    api_key_string = f"{endpoint}:{client_id}:{client_secret}:{org}"
    return base64.b64encode(api_key_string.encode("utf-8")).decode("utf-8")


def test_decode_api_key():
    """Test that API keys are correctly encoded and decoded."""
    from flyte.remote._client.auth._auth_utils import decode_api_key

    # Use a simple endpoint without scheme to avoid colon conflicts
    endpoint = "test.flyte.example.com"
    client_id = "test_client_id"
    client_secret = "test_client_secret"
    org = "test-org"

    encoded_api_key = create_encoded_api_key(endpoint, client_id, client_secret, org)

    decoded_endpoint, decoded_client_id, decoded_client_secret, decoded_org = decode_api_key(encoded_api_key)

    assert decoded_endpoint == endpoint
    assert decoded_client_id == client_id
    assert decoded_client_secret == client_secret
    assert decoded_org == org


def test_init_from_api_key_with_explicit_parameter():
    """Test init_from_api_key with API key passed as parameter."""
    endpoint = "test.flyte.example.com"
    client_id = "test_client_id"
    client_secret = "test_client_secret"
    org = "test-org"

    encoded_api_key = create_encoded_api_key(endpoint, client_id, client_secret, org)

    # Mock the init.aio function to avoid actually connecting to a server
    with mock.patch("flyte._initialize.init.aio", new_callable=mock.AsyncMock) as mock_init:
        init_from_api_key(api_key=encoded_api_key, project="test-project", domain="test-domain")

        # Verify that init.aio was called with the correct parameters
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args.kwargs

        # sanitize_endpoint adds dns:/// prefix to simple hostnames
        assert call_kwargs["endpoint"] == f"dns:///{endpoint}"
        assert call_kwargs["api_key"] == encoded_api_key
        assert call_kwargs["client_id"] == client_id
        assert call_kwargs["client_credentials_secret"] == client_secret
        assert call_kwargs["org"] == org
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["domain"] == "test-domain"
        assert call_kwargs["auth_type"] == "ClientSecret"


def test_init_from_api_key_with_environment_variable():
    """Test init_from_api_key reading API key from FLYTE_API_KEY environment variable."""
    endpoint = "test.flyte.example.com"
    client_id = "test_client_id"
    client_secret = "test_client_secret"
    org = "test-org"

    encoded_api_key = create_encoded_api_key(endpoint, client_id, client_secret, org)

    # Mock the environment variable
    with mock.patch.dict(os.environ, {"FLYTE_API_KEY": encoded_api_key}):
        # Mock the init.aio function to avoid actually connecting to a server
        with mock.patch("flyte._initialize.init.aio", new_callable=mock.AsyncMock) as mock_init:
            init_from_api_key(project="test-project", domain="test-domain")

            # Verify that init.aio was called with the correct parameters
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args.kwargs

            # sanitize_endpoint adds dns:/// prefix to simple hostnames
            assert call_kwargs["endpoint"] == f"dns:///{endpoint}"
            assert call_kwargs["api_key"] == encoded_api_key
            assert call_kwargs["client_id"] == client_id
            assert call_kwargs["client_credentials_secret"] == client_secret
            assert call_kwargs["org"] == org
            assert call_kwargs["project"] == "test-project"
            assert call_kwargs["domain"] == "test-domain"


def test_init_from_api_key_missing_api_key():
    """Test that init_from_api_key raises error when no API key is provided."""
    # Ensure FLYTE_API_KEY is not set
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(InitializationError) as exc_info:
            init_from_api_key(project="test-project", domain="test-domain")

        # Check that the error message contains the expected information
        error_message = str(exc_info.value)
        assert "API key must be provided" in error_message or "FLYTE_API_KEY" in error_message


def test_init_from_api_key_with_none_org():
    """Test init_from_api_key handles 'None' org string correctly."""
    endpoint = "test.flyte.example.com"
    client_id = "test_client_id"
    client_secret = "test_client_secret"
    org = ""  # Empty org

    encoded_api_key = create_encoded_api_key(endpoint, client_id, client_secret, org)

    # Mock the init.aio function
    with mock.patch("flyte._initialize.init.aio", new_callable=mock.AsyncMock) as mock_init:
        init_from_api_key(api_key=encoded_api_key, project="test-project", domain="test-domain")

        # Verify that org is None (not "None" string)
        call_kwargs = mock_init.call_args.kwargs
        assert call_kwargs["org"] is None


def test_init_from_api_key_parameter_override():
    """Test that init_from_api_key uses provided parameters correctly."""
    endpoint = "test.flyte.example.com"
    client_id = "test_client_id"
    client_secret = "test_client_secret"
    org = "test-org"

    encoded_api_key = create_encoded_api_key(endpoint, client_id, client_secret, org)

    # Mock the init.aio function
    with mock.patch("flyte._initialize.init.aio", new_callable=mock.AsyncMock) as mock_init:
        from pathlib import Path

        init_from_api_key(
            api_key=encoded_api_key,
            project="custom-project",
            domain="custom-domain",
            root_dir=Path("/custom/root"),
            batch_size=5000,
            image_builder="remote",
        )

        # Verify parameters were passed through
        call_kwargs = mock_init.call_args.kwargs
        assert call_kwargs["project"] == "custom-project"
        assert call_kwargs["domain"] == "custom-domain"
        assert call_kwargs["root_dir"] == Path("/custom/root")
        assert call_kwargs["batch_size"] == 5000
        assert call_kwargs["image_builder"] == "remote"
