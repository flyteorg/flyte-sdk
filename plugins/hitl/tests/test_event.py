"""Tests for HITL Event class and related functions."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flyteplugins.hitl._event import Event, EventFormLink


class TestEvent:
    """Tests for the Event class."""

    def test_event_form_url(self):
        """Test Event form_url property builds correct URL with encoded params."""
        event = Event(
            name="test_event",
            scope="run",
            data_type=str,
            prompt="Enter value",
            request_id="req-456",
            endpoint="https://hitl.example.com",
            request_path="/path/to/request.json",
            response_path="/path/to/response.json",
        )

        form_url = event.form_url

        assert "https://hitl.example.com/form/req-456" in form_url
        assert "request_path=" in form_url
        assert "%2Fpath%2Fto%2Frequest.json" in form_url


class TestEventCreate:
    """Tests for Event.create() class method."""

    @pytest.fixture(autouse=True)
    def reset_event_state(self):
        """Reset Event class state before each test."""
        Event._app_served = False
        Event._app_handle = None
        yield
        Event._app_served = False
        Event._app_handle = None

    @patch("flyteplugins.hitl._event.storage")
    @patch("flyteplugins.hitl._event._get_response_path")
    @patch("flyteplugins.hitl._event._get_request_path")
    @patch("flyteplugins.hitl._event.Event._serve_app")
    def test_create_serves_app_if_not_served(
        self,
        mock_serve_app,
        mock_get_request_path,
        mock_get_response_path,
        mock_storage,
    ):
        """Test that create() serves the app if not already served."""
        mock_app_handle = MagicMock()
        mock_app_handle.endpoint = "https://hitl.example.com"
        mock_serve_app.return_value = mock_app_handle

        mock_get_request_path.return_value = "/path/to/request.json"
        mock_get_response_path.return_value = "/path/to/response.json"
        mock_storage.put_stream = AsyncMock()

        event = Event.create(
            name="test_event",
            data_type=int,
            scope="run",
            prompt="Enter a number",
        )

        mock_serve_app.assert_called_once()
        assert Event._app_served is True
        assert event.name == "test_event"

    @patch("flyteplugins.hitl._event.storage")
    @patch("flyteplugins.hitl._event._get_response_path")
    @patch("flyteplugins.hitl._event._get_request_path")
    @patch("flyteplugins.hitl._event.Event._serve_app")
    def test_create_does_not_serve_app_if_already_served(
        self,
        mock_serve_app,
        mock_get_request_path,
        mock_get_response_path,
        mock_storage,
    ):
        """Test that create() does not serve the app if already served."""
        mock_app_handle = MagicMock()
        mock_app_handle.endpoint = "https://hitl.example.com"
        Event._app_served = True
        Event._app_handle = mock_app_handle

        mock_get_request_path.return_value = "/path/to/request.json"
        mock_get_response_path.return_value = "/path/to/response.json"
        mock_storage.put_stream = AsyncMock()

        Event.create(
            name="test_event",
            data_type=str,
            scope="run",
            prompt="Enter value",
        )

        mock_serve_app.assert_not_called()

    @patch("flyteplugins.hitl._event.storage")
    @patch("flyteplugins.hitl._event._get_response_path")
    @patch("flyteplugins.hitl._event._get_request_path")
    @patch("flyteplugins.hitl._event.Event._serve_app")
    def test_create_writes_request_metadata(
        self,
        mock_serve_app,
        mock_get_request_path,
        mock_get_response_path,
        mock_storage,
    ):
        """Test that create() writes request metadata to storage."""
        mock_app_handle = MagicMock()
        mock_app_handle.endpoint = "https://hitl.example.com"
        Event._app_served = True
        Event._app_handle = mock_app_handle

        mock_get_request_path.return_value = "/path/to/request.json"
        mock_get_response_path.return_value = "/path/to/response.json"
        mock_storage.put_stream = AsyncMock()

        Event.create(
            name="my_event",
            data_type=float,
            scope="run",
            prompt="Enter amount",
        )

        mock_storage.put_stream.assert_called_once()
        call_args = mock_storage.put_stream.call_args
        request_data = json.loads(call_args[0][0].decode())

        assert request_data["event_name"] == "my_event"
        assert request_data["prompt"] == "Enter amount"
        assert request_data["data_type"] == "float"
        assert request_data["status"] == "pending"
        assert "request_id" in request_data


class TestEventFormLink:
    """Tests for the EventFormLink class."""

    def test_event_form_link_get_link(self):
        """Test EventFormLink.get_link() method builds correct URL."""
        link = EventFormLink(
            endpoint="https://hitl.example.com",
            request_id="req-789",
            request_path="/path/to/request.json",
        )

        result = link.get_link(
            run_name="my-run",
            project="my-project",
            domain="development",
            context={},
            parent_action_name="parent",
            action_name="action",
            pod_name="pod-123",
        )

        assert "https://hitl.example.com/form/req-789" in result
        assert "request_path=" in result


class TestWaitForInputEvent:
    """Tests for the wait_for_input_event function."""

    @pytest.mark.asyncio
    @patch("flyteplugins.hitl._event.flyte.durable.sleep")
    @patch("flyteplugins.hitl._event.storage")
    async def test_wait_for_input_returns_value_when_response_exists(
        self,
        mock_storage,
        mock_durable_sleep,
    ):
        """Test that wait returns value when response is found."""
        from flyteplugins.hitl._event import wait_for_input_event

        response_data = {"status": "completed", "value": 42}

        async def mock_get_stream(path):
            yield json.dumps(response_data).encode()

        mock_storage.exists = AsyncMock(return_value=True)
        mock_storage.get_stream = mock_get_stream

        result = await wait_for_input_event(
            name="test_event",
            request_id="req-123",
            response_path="/path/to/response.json",
            timeout_seconds=60,
            poll_interval_seconds=5,
        )

        assert result == 42

    @pytest.mark.asyncio
    @patch("flyteplugins.hitl._event.flyte.durable.sleep")
    @patch("flyteplugins.hitl._event.storage")
    async def test_wait_for_input_polls_until_response(
        self,
        mock_storage,
        mock_durable_sleep,
    ):
        """Test that wait polls until response is found."""
        from flyteplugins.hitl._event import wait_for_input_event

        call_count = 0

        async def mock_exists(path):
            nonlocal call_count
            call_count += 1
            return call_count >= 3

        response_data = {"status": "completed", "value": "hello"}

        async def mock_get_stream(path):
            yield json.dumps(response_data).encode()

        mock_storage.exists = mock_exists
        mock_storage.get_stream = mock_get_stream
        mock_durable_sleep.aio = AsyncMock()

        result = await wait_for_input_event(
            name="test_event",
            request_id="req-123",
            response_path="/path/to/response.json",
            timeout_seconds=60,
            poll_interval_seconds=5,
        )

        assert result == "hello"
        assert mock_durable_sleep.aio.call_count == 2

    @pytest.mark.asyncio
    @patch("flyteplugins.hitl._event.flyte.durable.sleep")
    @patch("flyteplugins.hitl._event.storage")
    async def test_wait_for_input_raises_timeout_error(
        self,
        mock_storage,
        mock_durable_sleep,
    ):
        """Test that wait raises TimeoutError when timeout is reached."""
        from flyteplugins.hitl._event import wait_for_input_event

        mock_storage.exists = AsyncMock(return_value=False)
        mock_durable_sleep.aio = AsyncMock()

        with pytest.raises(TimeoutError) as exc_info:
            await wait_for_input_event(
                name="test_event",
                request_id="req-123",
                response_path="/path/to/response.json",
                timeout_seconds=10,
                poll_interval_seconds=5,
            )

        assert "test_event" in str(exc_info.value)
        assert "req-123" in str(exc_info.value)
