"""Regular package test - my_pkg.tasks.foo"""

from my_pkg.tasks import shared


def hello_regular():
    return f"Hello from regular package: {shared.get_message()}"


def task_regular_foo():
    """Task in regular package with absolute import"""
    return "regular task_foo executed"


# Loading message
print(f"In 3a_regular_package example: {__file__} module")
