import io
import logging
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
    @mock.patch("flyte._excepthook.traceback.print_tb")
    def test_filtered_traceback_without_cause(self, mock_print_tb, mock_extract_tb, mock_stdout):
        """Test filtered traceback printing without exception cause."""
        # Setup mock frames
        included_frame = mock.Mock()
        included_frame.name = "regular_function"
        included_frame.filename = "/path/to/regular_file.py"

        excluded_frame = mock.Mock()
        excluded_frame.name = "_internal_function"
        excluded_frame.filename = "/path/to/file.py"

        mock_extract_tb.return_value = [included_frame, excluded_frame]

        with mock.patch.object(logger, "getEffectiveLevel", return_value=logging.INFO):
            exc_type = ValueError
            exc_value = ValueError("test error")
            exc_tb = mock.Mock()

            custom_excepthook(exc_type, exc_value, exc_tb)

            # Verify traceback.print_tb was called with filtered frames
            mock_print_tb.assert_called_once_with([included_frame])

            # Check output contains expected content
            output = mock_stdout.getvalue()
            assert "Filtered traceback (most recent call last):" in output
            assert "ValueError: test error" in output

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    @mock.patch("flyte._excepthook.traceback.extract_tb")
    @mock.patch("flyte._excepthook.traceback.print_tb")
    def test_filtered_traceback_with_cause(self, mock_print_tb, mock_extract_tb, mock_stdout):
        """Test filtered traceback printing with exception cause."""
        # Setup mock frames
        included_frame = mock.Mock()
        included_frame.name = "regular_function"
        included_frame.filename = "/path/to/regular_file.py"

        mock_extract_tb.return_value = [included_frame]

        # Create exception with cause
        cause_exception = RuntimeError("root cause")
        main_exception = ValueError("test error")
        main_exception.__cause__ = cause_exception

        with mock.patch.object(logger, "getEffectiveLevel", return_value=logging.INFO):
            exc_type = ValueError
            exc_value = main_exception
            exc_tb = mock.Mock()

            custom_excepthook(exc_type, exc_value, exc_tb)

            # Verify traceback.print_tb was called
            mock_print_tb.assert_called_once_with([included_frame])

            # Check output contains expected content including cause
            output = mock_stdout.getvalue()
            assert "Filtered traceback (most recent call last):" in output
            assert "ValueError: test error" in output
            assert "Caused by RuntimeError: root cause" in output

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    @mock.patch("flyte._excepthook.traceback.extract_tb")
    @mock.patch("flyte._excepthook.traceback.print_tb")
    def test_filtered_traceback_without_cause_none_cause(self, mock_print_tb, mock_extract_tb, mock_stdout):
        """Test filtered traceback printing when exception cause is None."""
        # Setup mock frames
        included_frame = mock.Mock()
        included_frame.name = "regular_function"
        included_frame.filename = "/path/to/regular_file.py"

        mock_extract_tb.return_value = [included_frame]

        with mock.patch.object(logger, "getEffectiveLevel", return_value=logging.INFO):
            exc_type = ValueError
            exc_value = ValueError("test error")
            exc_value.__cause__ = None
            exc_tb = mock.Mock()

            custom_excepthook(exc_type, exc_value, exc_tb)

            # Verify traceback.print_tb was called
            mock_print_tb.assert_called_once_with([included_frame])

            # Check output does not contain cause information
            output = mock_stdout.getvalue()
            assert "Filtered traceback (most recent call last):" in output
            assert "ValueError: test error" in output
            assert "Caused by" not in output

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    @mock.patch("flyte._excepthook.traceback.extract_tb")
    @mock.patch("flyte._excepthook.traceback.print_tb")
    def test_all_frames_filtered_out(self, mock_print_tb, mock_extract_tb, mock_stdout):
        """Test behavior when all frames are filtered out."""
        # Setup mock frames that should all be excluded
        excluded_frame1 = mock.Mock()
        excluded_frame1.name = "_internal_function"
        excluded_frame1.filename = "/path/to/file.py"

        excluded_frame2 = mock.Mock()
        excluded_frame2.name = "regular_function"
        excluded_frame2.filename = "/path/to/syncify_file.py"

        mock_extract_tb.return_value = [excluded_frame1, excluded_frame2]

        with mock.patch.object(logger, "getEffectiveLevel", return_value=logging.INFO):
            exc_type = ValueError
            exc_value = ValueError("test error")
            exc_tb = mock.Mock()

            custom_excepthook(exc_type, exc_value, exc_tb)

            # Verify traceback.print_tb was called with empty list
            mock_print_tb.assert_called_once_with([])

            # Check output still contains error message
            output = mock_stdout.getvalue()
            assert "Filtered traceback (most recent call last):" in output
            assert "ValueError: test error" in output
