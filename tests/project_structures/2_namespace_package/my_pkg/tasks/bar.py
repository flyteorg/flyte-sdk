"""Namespace package test - my_pkg.tasks.bar"""


def hello_from_namespace_bar():
    return "Hello from my_pkg.tasks.bar"


def task_namespace_bar():
    """Another task in namespace package"""
    return "namespace task_bar executed"

# Loading message
print(f"In 2_namespace_package example: {__file__} module")
