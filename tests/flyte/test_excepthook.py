import io
import logging
import sys
import traceback
from unittest import mock

from flyte._excepthook import custom_excepthook, should_include_frame
from flyte._logging import logger


class TestShouldIncludeFrame:
    def test_should_include_frame_excludes_internal_modules(self):
        frame = mock.Mock()
        frame.name = "_internal_function"
        frame.filename = "/path/to/file.py"
        assert not should_include_frame(frame)

    def test_should_include_frame_excludes_syncify_modules(self):
        frame = mock.Mock()
        frame.name = "syncify_function"
        frame.filename = "/path/to/file.py"
        assert not should_include_frame(frame)

    def test_should_include_frame_excludes_syncify_files(self):
        frame = mock.Mock()
        frame.name = "regular_function"
        frame.filename = "/path/to/syncify_file.py"
        assert not should_include_frame(frame)

    def test_should_include_frame_excludes_code_bundle_files(self):
        frame = mock.Mock()
        frame.name = "regular_function"
        frame.filename = "/path/to/_code_bundle_file.py"
        assert not should_include_frame(frame)

    def test_should_include_frame_includes_regular_frames(self):
        frame = mock.Mock()
        frame.name = "regular_function"
        frame.filename = "/path/to/regular_file.py"
        assert should_include_frame(frame)


class TestCustomExcepthook:
    def test_uses_original_excepthook_when_debug_level(self):
        """Test that original excepthook is used when logger level is DEBUG or lower."""
        with mock.patch("flyte._excepthook.original_excepthook") as mock_original:
            with mock.patch.object(logger, "getEffectiveLevel", return_value=logging.DEBUG):
                exc_type = ValueError
                exc_value = ValueError("test error")
                exc_tb = None

                custom_excepthook(exc_type, exc_value, exc_tb)

                mock_original.assert_called_once_with(exc_type, exc_value, exc_tb)

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    @mock.patch("flyte._excepthook.traceback.extract_tb")
    @mock.patch("flyte._excepthook.traceback.print_list")
    def test_filtered_traceback_without_cause(self, mock_print_list, mock_extract_tb, mock_stdout):
        """Test filtered traceback printing without exception cause."""
        # Setup mock frames using FrameSummary

        included_frame = traceback.FrameSummary("/path/to/regular_file.py", 10, "regular_function")
        excluded_frame = traceback.FrameSummary("/path/to/file.py", 20, "_internal_function")

        mock_extract_tb.return_value = [included_frame, excluded_frame]

        with mock.patch.object(logger, "getEffectiveLevel", return_value=logging.INFO):
            exc_type = ValueError
            exc_value = ValueError("test error")
            exc_tb = mock.Mock()

            custom_excepthook(exc_type, exc_value, exc_tb)

            # Verify traceback.print_list was called with filtered frames
            mock_print_list.assert_called_once_with([included_frame])

            # Check output contains expected content
            output = mock_stdout.getvalue()
            assert "Filtered traceback (most recent call last):" in output
            assert "ValueError: test error" in output

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    @mock.patch("flyte._excepthook.traceback.extract_tb")
    @mock.patch("flyte._excepthook.traceback.print_list")
    def test_filtered_traceback_without_cause_none_cause(self, mock_print_list, mock_extract_tb, mock_stdout):
        """Test filtered traceback printing when exception cause is None."""
        # Setup mock frames using FrameSummary

        included_frame = traceback.FrameSummary("/path/to/regular_file.py", 10, "regular_function")

        mock_extract_tb.return_value = [included_frame]

        with mock.patch.object(logger, "getEffectiveLevel", return_value=logging.INFO):
            exc_type = ValueError
            exc_value = ValueError("test error")
            exc_value.__cause__ = None
            exc_tb = mock.Mock()

            custom_excepthook(exc_type, exc_value, exc_tb)

            # Verify traceback.print_list was called
            mock_print_list.assert_called_once_with([included_frame])

            # Check output does not contain cause information
            output = mock_stdout.getvalue()
            assert "Filtered traceback (most recent call last):" in output
            assert "ValueError: test error" in output
            assert "Caused by" not in output

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    @mock.patch("flyte._excepthook.traceback.extract_tb")
    @mock.patch("flyte._excepthook.traceback.print_list")
    def test_all_frames_filtered_out(self, mock_print_list, mock_extract_tb, mock_stdout):
        """Test behavior when all frames are filtered out."""
        # Setup mock frames that should all be excluded using FrameSummary

        excluded_frame1 = traceback.FrameSummary("/path/to/file.py", 10, "_internal_function")
        excluded_frame2 = traceback.FrameSummary("/path/to/syncify_file.py", 20, "regular_function")

        mock_extract_tb.return_value = [excluded_frame1, excluded_frame2]

        with mock.patch.object(logger, "getEffectiveLevel", return_value=logging.INFO):
            exc_type = ValueError
            exc_value = ValueError("test error")
            exc_tb = mock.Mock()

            custom_excepthook(exc_type, exc_value, exc_tb)

            # Verify traceback.print_list was called with empty list
            mock_print_list.assert_called_once_with([])

            # Check output still contains error message
            output = mock_stdout.getvalue()
            assert "Filtered traceback (most recent call last):" in output
            assert "ValueError: test error" in output

    def test_custom_excepthook_real_exception(self):
        """Integration test with a real exception and traceback - no mocking.

        This test demonstrates the complete flow: it generates a real exception,
        captures the output from custom_excepthook, and verifies the formatting.
        """

        # Capture stdout
        captured_output = io.StringIO()

        # Set logger to INFO level to trigger filtering
        original_level = logger.level
        logger.setLevel(logging.INFO)

        try:
            # Generate a real exception with traceback by creating actual frames
            try:
                # This will create a traceback with actual frames
                exec("def normal_func():\n    raise ValueError('real test error')\nnormal_func()")
            except ValueError:
                exc_type, exc_value, exc_tb = sys.exc_info()

                # Redirect stdout
                original_stdout = sys.stdout
                sys.stdout = captured_output

                try:
                    custom_excepthook(exc_type, exc_value, exc_tb)
                finally:
                    sys.stdout = original_stdout

            # Verify output
            output = captured_output.getvalue()

            # Should have the header
            assert "Filtered traceback (most recent call last):" in output

            # Should have the error message with correct exception type and message
            assert "ValueError: real test error" in output

            # Verify the output ends with a newline (proper formatting)
            assert output.endswith("\n")

        finally:
            logger.setLevel(original_level)
