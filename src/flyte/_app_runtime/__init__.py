"""
App runtime module

This module is designed for minimal dependencies and module loads, so that runtime can remain as fast as we can keep it.
We should ideally move it to the app module and make the app module, really fast to load!

TODO we might want to expose this as non private module.

"""

from inputs import get_input

__all__ = [
    "get_input",
]
