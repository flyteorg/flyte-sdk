"""Helpers for use inside Jupyter notebooks executed by NotebookTask.

These functions are meant to be called from within a notebook that is
being run as a Flyte task via papermill.
"""

from __future__ import annotations

from typing import Any


def record_outputs(**kwargs: Any) -> str:
    """Record output values from a notebook for use by downstream Flyte tasks.

    Call this as the **last expression** in a cell tagged ``"outputs"``.
    The returned protobuf text is captured by Jupyter as the cell output
    and later extracted by ``NotebookTask``.

    Values are serialized as Flyte Literals, so any type supported by
    Flyte's type system works — primitives, ``File``, ``Dir``,
    ``DataFrame``, dataclasses, etc.

    Example (cell tagged ``"outputs"``)::

        from flyteplugins.papermill import record_outputs

        record_outputs(result=42, summary="done")

    Args:
        **kwargs: Output name/value pairs.

    Returns:
        Protobuf text representation of a ``LiteralMap``. Jupyter captures
        this as the cell's text/plain output.
    """
    from flyte.syncify import syncify
    from flyte.types import TypeEngine
    from google.protobuf import text_format

    @syncify
    async def _to_literal_map(values: dict[str, Any]):
        from flyteidl2.core.literals_pb2 import LiteralMap

        literals = {}
        for name, val in values.items():
            py_type = type(val)
            lit_type = TypeEngine.to_literal_type(py_type)
            literals[name] = await TypeEngine.to_literal(val, py_type, lit_type)
        return LiteralMap(literals=literals)

    literal_map = _to_literal_map(kwargs)
    return text_format.MessageToString(literal_map)


def load_file(path: str):
    """Load a ``flyte.io.File`` from a serialized path inside a notebook.

    When a ``File`` is passed as an input to a ``NotebookTask``, it is
    serialized to its remote path string for papermill injection.  Use
    this helper to reconstruct the ``File`` object inside the notebook::

        from flyteplugins.papermill import load_file

        f = load_file(my_file_path)  # my_file_path injected by papermill
        with f.open_sync() as fh:
            data = fh.read()

    Args:
        path: The remote path string (injected as a papermill parameter).

    Returns:
        A ``flyte.io.File`` instance pointing at the remote path.
    """
    from flyte.io import File

    return File(path=path)


def load_dir(path: str):
    """Load a ``flyte.io.Dir`` from a serialized path inside a notebook.

    When a ``Dir`` is passed as an input to a ``NotebookTask``, it is
    serialized to its remote path string.  Use this helper to
    reconstruct it::

        from flyteplugins.papermill import load_dir

        d = load_dir(my_dir_path)

    Args:
        path: The remote path string (injected as a papermill parameter).

    Returns:
        A ``flyte.io.Dir`` instance pointing at the remote path.
    """
    from flyte.io import Dir

    return Dir(path=path)


def load_dataframe(uri: str, fmt: str = "parquet"):
    """Load a ``flyte.io.DataFrame`` from a serialized URI inside a notebook.

    When a ``DataFrame`` is passed as an input to a ``NotebookTask``, it is
    serialized to its remote URI for papermill injection.  Use this helper
    to reconstruct it::

        from flyteplugins.papermill import load_dataframe

        df = load_dataframe(my_df_uri)
        pandas_df = df.all()  # materializes as pandas DataFrame

    Args:
        uri: The remote URI string (injected as a papermill parameter).
        fmt: The storage format (default ``"parquet"``).

    Returns:
        A ``flyte.io.DataFrame`` instance pointing at the remote URI.
    """
    from flyte.io import DataFrame

    return DataFrame(uri=uri, format=fmt)
