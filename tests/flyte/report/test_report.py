from unittest.mock import patch

import pytest
from mock.mock import AsyncMock

from flyte._context import internal_ctx
from flyte.models import ActionID, RawDataPath, TaskContext
from flyte.report import current_report, get_tab, log, replace
from flyte.report._report import Report


@pytest.fixture
def report():
    # Fixture to provide a fresh report for each test
    with internal_ctx().replace_task_context(
        TaskContext(
            action=ActionID(name="test"),
            report=Report("test_report"),
            version="test",
            raw_data_path=RawDataPath(path="test"),
            output_path="test",
            run_base_dir="test",
        )
    ):
        yield current_report()


def test_log_to_main_tab(report):
    # Log content to the main tab
    log("Test content")
    main_tab = report.get_tab("main")
    assert "Test content" in main_tab.content


def test_replace_main_tab_content(report):
    # Replace content in the main tab
    replace("New content")
    main_tab = report.get_tab("main")
    assert main_tab.content == ["New content"]


@patch("flyte.report._report.flush", new_callable=AsyncMock)
def test_flush_main_tab(mock_flush, report):
    # Log content and flush the report
    log("Flush content", do_flush=True)
    mock_flush.aio.assert_called_once()


def test_add_new_tab(report):
    # Add a new tab and log content to it
    new_tab = get_tab("new_tab")
    new_tab.log("New tab content")
    assert "new_tab" in report.tabs
    assert "New tab content" in report.get_tab("new_tab").content


def test_replace_new_tab_content(report):
    # Replace content in a new tab
    new_tab = get_tab("new_tab")
    new_tab.replace("Replaced content")
    assert new_tab.content == ["Replaced content"]


def test_get_final_report(report):
    # Test the final report generation
    log("Main tab content")
    new_tab = get_tab("new_tab")
    new_tab.log("New tab content")
    final_report = report.get_final_report()
    assert "Main tab content" in final_report
    assert "New tab content" in final_report


@patch("flyte._internal.runtime.io.report_path")
@patch("flyte.storage.put_stream", new_callable=AsyncMock)
def test_flush_saves_report(mock_put_stream, mock_report_path, report):
    # Mock the flush process to ensure it saves the report
    mock_report_path.return_value = "/mock/path/report.html"
    mock_put_stream.return_value = "/mock/path/report.html"

    log("Content to flush", do_flush=True)
    mock_put_stream.assert_called_once()
    args, _ = mock_put_stream.call_args
    assert b"Content to flush" in args[0]  # Check if content is in the flushed data


def test_fresh_report_has_no_content():
    """__post_init__ always creates the "main" tab, so existence != content."""
    assert Report("empty").has_content() is False


def test_empty_tab_alone_is_not_content():
    report = Report("r")
    report.get_tab("A")
    assert report.has_content() is False


def test_logged_main_tab_is_content():
    report = Report("r")
    report.get_tab("main").log("<p>hello</p>")
    assert report.has_content() is True


def test_logged_named_tab_is_content():
    """A task may log only to a named tab, e.g. Deck("Frame Renderer", ...)."""
    report = Report("r")
    report.get_tab("Frame Renderer").log("<p>profile</p>")
    assert report.has_content() is True
