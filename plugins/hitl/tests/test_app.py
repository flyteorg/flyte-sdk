"""Tests for HITL FastAPI app endpoints."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from flyteplugins.hitl._app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestInputForm:
    """Tests for the input form endpoint."""

    @patch("flyteplugins.hitl._app.storage")
    @patch("flyteplugins.hitl._app._get_request_path")
    def test_input_form_without_request_path(self, mock_get_request_path, mock_storage, client):
        """Test input form without request_path falls back to local path."""
        mock_get_request_path.return_value = "/tmp/hitl/request.json"
        mock_storage.exists = AsyncMock(return_value=False)

        response = client.get("/form/test-request-id")

        assert response.status_code == 200
        mock_get_request_path.assert_called_once_with("test-request-id")

    @patch("flyteplugins.hitl._app.aiofiles")
    @patch("flyteplugins.hitl._app.storage")
    def test_input_form_loads_request_metadata(self, mock_storage, mock_aiofiles, client):
        """Test input form loads and displays request metadata."""
        mock_storage.exists = AsyncMock(return_value=True)
        mock_storage.get = AsyncMock(return_value="/tmp/request.json")

        request_data = {
            "prompt": "What is your answer?",
            "data_type": "int",
            "event_name": "my_event",
            "response_path": "/path/to/response.json",
        }

        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=json.dumps(request_data))
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=None)
        mock_aiofiles.open = MagicMock(return_value=mock_file)

        response = client.get("/form/test-id?request_path=/path/to/request.json")

        assert response.status_code == 200
        assert "my_event" in response.text
        assert "What is your answer?" in response.text

    @patch("flyteplugins.hitl._app.storage")
    def test_input_form_handles_storage_error(self, mock_storage, client):
        """Test input form handles storage errors gracefully."""
        mock_storage.exists = AsyncMock(side_effect=Exception("Storage error"))

        response = client.get("/form/test-id?request_path=/path/to/request.json")

        assert response.status_code == 200
        assert "test-id" in response.text


class TestSubmitInput:
    """Tests for the form submission endpoint."""

    @patch("flyteplugins.hitl._app.storage")
    def test_submit_input_success(self, mock_storage, client):
        """Test successful form submission with type conversion."""
        mock_storage.put_stream = AsyncMock()

        response = client.post(
            "/submit",
            data={
                "request_id": "req-123",
                "value": "42",
                "data_type": "int",
                "response_path": "/path/to/response.json",
            },
        )

        assert response.status_code == 200
        assert "Submission Successful" in response.text
        call_args = mock_storage.put_stream.call_args
        response_data = json.loads(call_args[0][0].decode())
        assert response_data["value"] == 42
        assert isinstance(response_data["value"], int)

    def test_submit_input_invalid_int(self, client):
        """Test that invalid int value returns 400 error."""
        response = client.post(
            "/submit",
            data={
                "request_id": "req-123",
                "value": "not_a_number",
                "data_type": "int",
                "response_path": "/path/to/response.json",
            },
        )

        assert response.status_code == 400
        assert "Failed to convert value" in response.json()["detail"]

    @patch("flyteplugins.hitl._app.storage")
    @patch("flyteplugins.hitl._app._get_response_path")
    def test_submit_input_without_response_path(self, mock_get_response_path, mock_storage, client):
        """Test submission without response_path falls back to local path."""
        mock_get_response_path.return_value = "/tmp/hitl/response.json"
        mock_storage.put_stream = AsyncMock()

        response = client.post(
            "/submit",
            data={
                "request_id": "req-123",
                "value": "test",
                "data_type": "str",
            },
        )

        assert response.status_code == 200
        mock_get_response_path.assert_called_once_with("req-123")

    @patch("flyteplugins.hitl._app.storage")
    def test_submit_input_storage_error(self, mock_storage, client):
        """Test that storage errors return 500 error."""
        mock_storage.put_stream = AsyncMock(side_effect=Exception("Storage write failed"))

        response = client.post(
            "/submit",
            data={
                "request_id": "req-123",
                "value": "test",
                "data_type": "str",
                "response_path": "/path/to/response.json",
            },
        )

        assert response.status_code == 500
        assert "Failed to save response" in response.json()["detail"]


class TestSubmitInputJson:
    """Tests for the JSON submission endpoint."""

    @patch("flyteplugins.hitl._app.storage")
    def test_submit_json_success(self, mock_storage, client):
        """Test successful JSON submission."""
        mock_storage.put_stream = AsyncMock()

        response = client.post(
            "/submit/json",
            json={
                "request_id": "req-123",
                "value": 42,
                "data_type": "int",
                "response_path": "/path/to/response.json",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "submitted"
        assert data["request_id"] == "req-123"
        assert data["value"] == 42

    def test_submit_json_invalid_value(self, client):
        """Test that invalid value returns 400 error."""
        response = client.post(
            "/submit/json",
            json={
                "request_id": "req-123",
                "value": "not_a_number",
                "data_type": "int",
                "response_path": "/path/to/response.json",
            },
        )

        assert response.status_code == 400
        assert "Failed to convert value" in response.json()["detail"]

    @patch("flyteplugins.hitl._app.storage")
    def test_submit_json_storage_error(self, mock_storage, client):
        """Test that storage errors return 500 error."""
        mock_storage.put_stream = AsyncMock(side_effect=Exception("Storage write failed"))

        response = client.post(
            "/submit/json",
            json={
                "request_id": "req-123",
                "value": "test",
                "data_type": "str",
                "response_path": "/path/to/response.json",
            },
        )

        assert response.status_code == 500
        assert "Failed to save response" in response.json()["detail"]


class TestGetStatus:
    """Tests for the status endpoint."""

    @patch("flyteplugins.hitl._app.storage")
    def test_get_status_pending(self, mock_storage, client):
        """Test status when request exists but no response."""
        request_data = {"request_id": "req-123", "prompt": "Enter value"}

        async def mock_get_stream(path):
            if "request" in path:
                yield json.dumps(request_data).encode()

        mock_storage.exists = AsyncMock(side_effect=lambda p: "request" in p)
        mock_storage.get_stream = mock_get_stream

        response = client.get(
            "/status/req-123",
            params={
                "request_path": "/path/to/request.json",
                "response_path": "/path/to/response.json",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["request"] == request_data
        assert data["response"] is None

    @patch("flyteplugins.hitl._app.storage")
    def test_get_status_completed(self, mock_storage, client):
        """Test status when both request and response exist."""
        request_data = {"request_id": "req-123", "prompt": "Enter value"}
        response_data = {"value": 42, "status": "completed"}

        async def mock_get_stream(path):
            if "request" in path:
                yield json.dumps(request_data).encode()
            elif "response" in path:
                yield json.dumps(response_data).encode()

        mock_storage.exists = AsyncMock(return_value=True)
        mock_storage.get_stream = mock_get_stream

        response = client.get(
            "/status/req-123",
            params={
                "request_path": "/path/to/request.json",
                "response_path": "/path/to/response.json",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["response"] == response_data

    @patch("flyteplugins.hitl._app.storage")
    def test_get_status_not_found(self, mock_storage, client):
        """Test status when request does not exist."""
        mock_storage.exists = AsyncMock(return_value=False)

        response = client.get(
            "/status/req-123",
            params={
                "request_path": "/path/to/request.json",
                "response_path": "/path/to/response.json",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_found"
        assert data["request"] is None
