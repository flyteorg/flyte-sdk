import inspect
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Documentation:
    """
    This class is used to store the documentation of a task.

    It can be set explicitly or extracted from the docstring of the task.
    """

    description: str

    def __help__str__(self):
        return self.description


def extract_docstring(func: Optional[Callable]) -> Documentation:
    """
    Extracts the description from a docstring.
    """
    if func is None:
        return Documentation(description="")
    docstring = inspect.getdoc(func)
    if not docstring:
        return Documentation(description="")
    return Documentation(description=docstring)
