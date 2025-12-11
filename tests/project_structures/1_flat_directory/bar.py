"""Flat directory test - bar.py"""


def hello_bar():
    return "Hello from bar"


def task_bar():
    """Simple task in bar"""
    return "task_bar executed"

# Loading message
print(f"In 1_flat_directory example: {__file__} module")
