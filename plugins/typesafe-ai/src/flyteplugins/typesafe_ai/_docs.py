"""Recover an enum's own documentation, so the criteria do not have to be repeated.

An enum already says what its members mean -- in a class docstring, and in the
string literals underneath each member. The first is a real attribute; the second
is not. A member assignment followed by a bare string literal leaves nothing on the
member at runtime (`Severity.NONE.__doc__` returns the *class* docstring it
inherits), so the only way to read it is to parse the source, which is what pydantic
does for `use_attribute_docstrings` too.

That makes member docs a best-effort input: they work from a normal module on
disk and quietly return nothing when the source is not available (a REPL, `exec`,
some frozen or zipped deployments). Everything here degrades to the member name
rather than failing, because a missing description is a worse prompt, not a broken
program.
"""

from __future__ import annotations

import ast
import enum
import inspect
import textwrap
from functools import lru_cache
from typing import Dict, Optional, Type

#: Python <= 3.11 writes this docstring onto an undocumented Enum class itself.
#: It is not something the author wrote, so it must not be mistaken for a question.
#: (3.12+ leaves __doc__ as None on the class and only inherits from Enum.)
_DEFAULT_ENUM_DOC = "An enumeration."


@lru_cache(maxsize=None)
def class_doc(cls: Type) -> Optional[str]:
    """The enum's *own* class docstring, or None if it does not really have one.

    Deliberately not `inspect.getdoc`: that inherits, and an undocumented enum
    would then report CPython's own text -- "Create a collection of name/value
    pairs." on 3.12+, or "Enum where members are also (and must be) ints" for an
    IntEnum -- which would be sent to the model as the question. Reading the class
    dict directly means only a docstring written on this enum counts.
    """
    doc = cls.__dict__.get("__doc__")
    if not isinstance(doc, str) or not doc.strip() or doc.strip() == _DEFAULT_ENUM_DOC:
        return None
    return " ".join(doc.split())


@lru_cache(maxsize=None)
def member_docs(cls: Type[enum.Enum]) -> Dict[str, str]:
    """{member name: docstring} for members documented by a literal underneath them."""
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):  # no source on disk: interactive, exec'd, frozen
        return {}
    try:
        tree = ast.parse(textwrap.dedent(source)).body[0]
    except (SyntaxError, IndexError):
        return {}
    if not isinstance(tree, ast.ClassDef):
        return {}

    docs: Dict[str, str] = {}
    pending: Optional[str] = None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            pending = target.id if isinstance(target, ast.Name) else None
        elif (
            pending
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            docs[pending] = " ".join(node.value.value.split())
            pending = None
        else:
            pending = None
    return docs
