"""Shared utilities for testing imports"""


def get_message():
    return "shared module loaded correctly"


# Loading message
print(f"In 3a_regular_package example: {__file__} module")
