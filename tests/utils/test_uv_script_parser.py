import pathlib
import sys

from flyte._image import CopyConfig, Image
from flyte._utils.uv_script_parser import UVScriptMetadata, parse_uv_script_file


def test_uv_script_parser(tmp_path: pathlib.Path):
    # Create a temporary file with a valid UV script
    script_content = """
# /// script
# dependencies = [
#    "polars",
#    "flyte @ file:${PROJECT_ROOT}",
# ]
# requires-python = ">=3.8"
# [tool.uv]
# exclude-newer = "2023-10-16T00:00:00Z"
# ///
import flyte
def main():
    pass
"""
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding="utf-8")

    # Parse the UV script file
    metadata = parse_uv_script_file(script_path)

    # Check the parsed metadata
    assert isinstance(metadata, UVScriptMetadata)
    assert metadata.requires_python == ">=3.8"
    assert metadata.dependencies == [
        "polars",
        "flyte @ file:${PROJECT_ROOT}",
    ]
    assert metadata.tool is not None
    assert metadata.tool["uv"].exclude_newer == "2023-10-16T00:00:00Z"


def test_identifier(tmp_path: pathlib.Path):
    # Create a temporary file with a valid UV script
    script_content = """
# /// script
# dependencies = [
#    "polars",
# ]
# requires-python = ">=3.8"
# [tool.uv]
# exclude-newer = "2023-10-16T00:00:00Z"
# ///
import flyte
def main():
    pass
"""
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding="utf-8")

    img = Image.from_uv_script(
        tmp_path / "test_script.py", registry="ghcr.io/wild-endeavor", name="test", python_version=(3, 12)
    )

    img = img.clone(
        addl_layer=CopyConfig(
            path_type=1,
            src=tmp_path,
            dst=".",
        )
    )

    # shouldn't change even though the temp path changes
    if sys.version_info == (3, 13):
        assert img.identifier == "iRLq7x1r4ewhrcX7VTxkzw"
