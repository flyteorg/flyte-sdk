"""Flat directory test - foo.py"""


def hello_foo():
    return "Hello from foo"


def task_foo():
    """Simple task in foo"""
    return "task_foo executed"

# Loading message
print(f"In 1_flat_directory example: {__file__} module")