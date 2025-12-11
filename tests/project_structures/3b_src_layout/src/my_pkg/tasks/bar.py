"""Src layout test - my_pkg.tasks.bar"""

from . import shared


def hello_src_bar():
    return f"Hello from src bar: {shared.get_src_message()}"


def task_src_bar():
    """Task in src layout with relative import"""
    return "src layout task_bar executed"


# Loading message
print(f"In 3b_src_layout example: {__file__} module")
