"""Tests for HITL HTML template functions."""

from flyteplugins.hitl._html_templates import (
    get_bool_select_element,
    get_event_report_html,
    get_index_page,
    get_input_form_page,
    get_submission_success_page,
)


class TestGetIndexPage:
    """Tests for get_index_page() template function."""

    def test_get_index_page_contains_endpoints(self):
        """Test that index page documents the available endpoints."""
        result = get_index_page()

        assert "Human-in-the-Loop Event Service" in result
        assert "/form/{request_id}" in result
        assert "/submit" in result
        assert "/status/{request_id}" in result


class TestGetBoolSelectElement:
    """Tests for get_bool_select_element() template function."""

    def test_get_bool_select_element_has_options(self):
        """Test that bool select has true/false options."""
        result = get_bool_select_element()

        assert '<option value="true">True</option>' in result
        assert '<option value="false">False</option>' in result
        assert '<option value="">-- Select --</option>' in result


class TestGetInputFormPage:
    """Tests for get_input_form_page() template function."""

    def test_get_input_form_page_contains_form_elements(self):
        """Test that input form page contains event info and form elements."""
        result = get_input_form_page(
            event_name="my_approval_event",
            prompt="Do you approve?",
            data_type_name="bool",
            request_id="req-456",
            response_path="/path/to/response.json",
            input_element='<select name="value"></select>',
        )

        assert "my_approval_event" in result
        assert "Do you approve?" in result
        assert "bool" in result
        assert "req-456" in result
        assert '<form action="/submit" method="post">' in result
        assert '<button type="submit">Submit</button>' in result


class TestGetSubmissionSuccessPage:
    """Tests for get_submission_success_page() template function."""

    def test_get_submission_success_page_escapes_html(self):
        """Test that submission success page escapes HTML in values."""
        result = get_submission_success_page(
            request_id="req-789",
            value="<script>alert('xss')</script>",
            data_type="str",
            message="Test <b>message</b>",
        )

        assert "&lt;script&gt;" in result
        assert "&lt;b&gt;" in result
        assert "<script>" not in result


class TestGetEventReportHtml:
    """Tests for get_event_report_html() template function."""

    def test_get_event_report_html_contains_urls_and_curl(self):
        """Test that event report contains form URL and curl example."""
        form_url = "https://hitl.example.com/form/my-request-id?request_path=/path"
        api_url = "https://hitl.example.com/submit/json"
        result = get_event_report_html(
            form_url=form_url,
            api_url=api_url,
            curl_body='{"request_id": "req-456", "value": 42}',
            type_name="int",
        )

        assert form_url in result
        assert api_url in result
        assert "curl" in result
        assert "Event Input Required" in result

    def test_get_event_report_html_escapes_html(self):
        """Test that event report escapes HTML in URLs and body."""
        result = get_event_report_html(
            form_url="https://example.com/form?param=<script>",
            api_url="https://example.com/submit",
            curl_body='{"value": "<script>alert(1)</script>"}',
            type_name="str",
        )

        assert "&lt;script&gt;" in result
        assert "<script>" not in result
