import subprocess
import sys
import textwrap


def test_guess_python_type_reverses_dataframe_without_prior_import():
    # flyte.io loads DataFrame lazily, so a process that reverses a remote task interface (for example
    # resolving a deployed task's signature) may never have imported it. Run in a fresh interpreter,
    # since other tests in this session import DataFrame and register its transformer.
    code = textwrap.dedent(
        """
        from flyteidl2.core import types_pb2
        from flyte.types import TypeEngine

        sd = types_pb2.LiteralType(structured_dataset_type=types_pb2.StructuredDatasetType())
        print(TypeEngine.guess_python_type(sd).__name__)
        print(TypeEngine.guess_python_type(types_pb2.LiteralType(collection_type=sd)))
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    lines = out.strip().splitlines()
    assert lines[0] == "DataFrame"
    assert "DataFrame" in lines[1] and "list" in lines[1].lower()
