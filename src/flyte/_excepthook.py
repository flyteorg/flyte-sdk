import logging
import sys
import traceback

from flyte._logging import logger

# Save the original excepthook so we can call it later
original_excepthook = sys.excepthook

# Filters: exclude frames where filename or function name contains these substrings
EXCLUDED_MODULE_SUBSTRINGS = ["_internal", "syncify"]
EXCLUDED_FILE_SUBSTRINGS = ["syncify", "_code_bundle"]


def should_include_frame(frame: traceback.FrameSummary) -> bool:
    return not (
        any(sub in frame.name for sub in EXCLUDED_MODULE_SUBSTRINGS)
        or any(sub in frame.filename for sub in EXCLUDED_FILE_SUBSTRINGS)
    )


def custom_excepthook(exc_type, exc_value, exc_tb):
    """
    Custom exception hook to filter and format tracebacks.
    If the logger's level is DEBUG, it uses the original excepthook.
    """

    if logger.getEffectiveLevel() <= logging.DEBUG:
        original_excepthook(exc_type, exc_value, exc_tb)
    else:
        # Extract and filter traceback
        tb_list = traceback.extract_tb(exc_tb)
        filtered_tb = [frame for frame in tb_list if should_include_frame(frame)]
        # Print the filtered version (custom format)
        print("Filtered traceback (most recent call last):")
        traceback.print_list(filtered_tb)
        print(f"{exc_type.__name__}: {exc_value}\n")
