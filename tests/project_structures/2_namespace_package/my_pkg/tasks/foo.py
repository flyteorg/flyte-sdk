"""Namespace package test - my_pkg.tasks.foo"""


def hello_from_namespace():
    return "Hello from my_pkg.tasks.foo"


def task_namespace_foo():
    """Task in namespace package"""
    return "namespace task_foo executed"


# Loading message
print(f"In 2_namespace_package example: {__file__} module")
