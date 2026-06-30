import functools

from flyte.types import TypeEngine

from ._jsonl_dir import JsonlDir
from ._jsonl_file import JsonlFile

__all__ = ["JsonlDir", "JsonlFile"]


@functools.lru_cache(maxsize=None)
def register_jsonl_type():
    """Register JsonlFile and JsonlDir with the Flyte type engine.

    This function is called automatically via the flyte.plugins.types entry point
    when flyte.init() is called with load_plugin_type_transformers=True (the default).

    JsonlFile is a File subclass, so it reuses the existing FileTransformer.
    JsonlDir is a Dir subclass, so it reuses the existing DirTransformer.
    """
    from flyte.io._dir import DirTransformer
    from flyte.io._file import FileTransformer

    TypeEngine.register_additional_type(FileTransformer(), JsonlFile)
    TypeEngine.register_additional_type(DirTransformer(), JsonlDir)


# Also register at module import time for backwards compatibility
register_jsonl_type()
