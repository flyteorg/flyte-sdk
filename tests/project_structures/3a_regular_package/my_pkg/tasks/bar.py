"""Regular package test - my_pkg.tasks.bar"""

from . import shared


def hello_regular_bar():
    return f"Hello from bar: {shared.get_message()}"


def task_regular_bar():
    """Task in regular package with relative import"""
    return "regular task_bar executed"
