"""Src layout test - my_pkg.tasks.foo"""

from my_pkg.tasks import shared


def hello_src_layout():
    return f"Hello from src layout: {shared.get_src_message()}"


def task_src_foo():
    """Task in src layout package"""
    return "src layout task_foo executed"


# Loading message
print(f"In 3b_src_layout example: {__file__} module")
