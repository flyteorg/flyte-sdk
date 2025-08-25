from typing import TypeVar

T = TypeVar("T")


class HashOnReferenceMixin(object):
    def __hash__(self):
        return hash(id(self))
