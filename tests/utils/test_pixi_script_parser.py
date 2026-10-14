import pathlib

import pytest
import toml

from flyte._image import Image, PixiScript
from flyte._utils.pixi_script_parser import (
    check_platforms_supported,
    parse_pixi_script_file,
    platforms_for,
    render_pixi_manifest,
)

FULL_SCRIPT = """\
#!/usr/bin/env -S pixi run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "flyte"]
#
# [tool.pixi.workspace]
# channels = ["conda-forge", "bioconda"]
#
# [tool.pixi.dependencies]
# gdal = "*"
#
# [tool.pixi.target.linux-64.dependencies]
# libblas = "*"
# ///
import httpx
"""


def write(tmp_path: pathlib.Path, content: str, name: str = "script.py") -> pathlib.Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_pixi_script(tmp_path: pathlib.Path):
    metadata = parse_pixi_script_file(write(tmp_path, FULL_SCRIPT))

    assert metadata.requires_python == ">=3.12"
    assert metadata.dependencies == ["httpx", "flyte"]
    # The whole [tool.pixi] table is kept verbatim.
    assert metadata.pixi["workspace"] == {"channels": ["conda-forge", "bioconda"]}
    assert metadata.pixi["dependencies"] == {"gdal": "*"}
    assert metadata.pixi["target"] == {"linux-64": {"dependencies": {"libblas": "*"}}}


def test_parse_pixi_script_without_metadata_block(tmp_path: pathlib.Path):
    with pytest.raises(ValueError, match="No PEP 723 script metadata block found"):
        parse_pixi_script_file(write(tmp_path, "import flyte\n", name="no_block.py"))


def test_parse_pixi_script_with_invalid_toml(tmp_path: pathlib.Path):
    script = "# /// script\n# dependencies = [oops\n# ///\n"
    with pytest.raises(ValueError, match="Invalid TOML"):
        parse_pixi_script_file(write(tmp_path, script, name="bad_toml.py"))


def test_parse_pixi_script_missing_file(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        parse_pixi_script_file(tmp_path / "nope.py")


def test_render_manifest(tmp_path: pathlib.Path):
    metadata = parse_pixi_script_file(write(tmp_path, FULL_SCRIPT))
    manifest = toml.loads(render_pixi_manifest(metadata, ("linux-64",)))

    # Channels come from the script; platforms are filled in from the image.
    assert manifest["workspace"] == {"channels": ["conda-forge", "bioconda"], "platforms": ["linux-64"]}
    # requires-python becomes the environment's python, alongside the conda dependencies.
    assert manifest["dependencies"] == {"gdal": "*", "python": ">=3.12"}
    # PEP 723 dependencies are PyPI packages.
    assert manifest["pypi-dependencies"] == {"httpx": "*", "flyte": "*"}
    # Tables flyte does not model are passed through untouched.
    assert manifest["target"] == {"linux-64": {"dependencies": {"libblas": "*"}}}


def test_render_manifest_defaults(tmp_path: pathlib.Path):
    """A script with no [tool.pixi] table at all still produces an installable manifest."""
    script = '# /// script\n# dependencies = ["cowsay"]\n# ///\nprint(1)\n'
    metadata = parse_pixi_script_file(write(tmp_path, script, name="bare.py"))
    manifest = toml.loads(render_pixi_manifest(metadata, ("linux-64", "linux-aarch64")))

    # pixi defaults a script to conda-forge; a workspace manifest must say so explicitly.
    assert manifest["workspace"] == {"channels": ["conda-forge"], "platforms": ["linux-64", "linux-aarch64"]}
    assert manifest["pypi-dependencies"] == {"cowsay": "*"}
    assert "dependencies" not in manifest


def test_render_manifest_script_platforms_win(tmp_path: pathlib.Path):
    script = (
        '# /// script\n# dependencies = ["cowsay"]\n#\n# [tool.pixi.workspace]\n# platforms = ["linux-64"]\n# ///\n'
    )
    metadata = parse_pixi_script_file(write(tmp_path, script, name="pinned.py"))
    manifest = toml.loads(render_pixi_manifest(metadata, ("linux-64",)))
    assert manifest["workspace"]["platforms"] == ["linux-64"]


def test_render_manifest_explicit_python_wins(tmp_path: pathlib.Path):
    """pixi documents [tool.pixi.dependencies].python as taking precedence over requires-python."""
    script = '# /// script\n# requires-python = ">=3.10"\n#\n# [tool.pixi.dependencies]\n# python = "3.12.*"\n# ///\n'
    metadata = parse_pixi_script_file(write(tmp_path, script, name="py.py"))
    manifest = toml.loads(render_pixi_manifest(metadata, ("linux-64",)))
    assert manifest["dependencies"]["python"] == "3.12.*"


def test_render_manifest_pypi_requirement_forms(tmp_path: pathlib.Path):
    script = (
        "# /// script\n"
        "# dependencies = [\n"
        '#   "httpx>=0.28,<1",\n'
        '#   "pandas[performance]",\n'
        '#   "mypkg @ https://example.com/mypkg-1.0-py3-none-any.whl",\n'
        '#   "othpkg @ git+https://github.com/org/othpkg@v1.2.3",\n'
        "# ]\n"
        "# ///\n"
    )
    metadata = parse_pixi_script_file(write(tmp_path, script, name="forms.py"))
    manifest = toml.loads(render_pixi_manifest(metadata, ("linux-64",)))

    pypi = manifest["pypi-dependencies"]
    # `packaging` normalizes specifier ordering, which keeps the rendered manifest stable.
    assert pypi["httpx"] == "<1,>=0.28"
    assert pypi["pandas"] == {"version": "*", "extras": ["performance"]}
    assert pypi["mypkg"] == {"url": "https://example.com/mypkg-1.0-py3-none-any.whl"}
    assert pypi["othpkg"] == {"git": "https://github.com/org/othpkg", "rev": "v1.2.3"}


def test_render_manifest_pypi_dependencies_table_wins(tmp_path: pathlib.Path):
    """[tool.pixi.pypi-dependencies] is the more specific declaration, so it is not overwritten."""
    script = (
        "# /// script\n"
        '# dependencies = ["httpx"]\n'
        "#\n"
        "# [tool.pixi.pypi-dependencies]\n"
        '# httpx = { version = ">=0.28", extras = ["socks"] }\n'
        "# ///\n"
    )
    metadata = parse_pixi_script_file(write(tmp_path, script, name="dup.py"))
    manifest = toml.loads(render_pixi_manifest(metadata, ("linux-64",)))
    assert manifest["pypi-dependencies"]["httpx"] == {"version": ">=0.28", "extras": ["socks"]}


def test_render_manifest_rejects_environment_markers(tmp_path: pathlib.Path):
    script = "# /// script\n# dependencies = [\"httpx; python_version < '3.11'\"]\n# ///\n"
    metadata = parse_pixi_script_file(write(tmp_path, script, name="marker.py"))
    with pytest.raises(ValueError, match="Environment markers are not supported"):
        render_pixi_manifest(metadata, ("linux-64",))


def test_render_manifest_rejects_invalid_requirement(tmp_path: pathlib.Path):
    script = '# /// script\n# dependencies = ["not a requirement!!"]\n# ///\n'
    metadata = parse_pixi_script_file(write(tmp_path, script, name="invalid.py"))
    with pytest.raises(ValueError, match="Invalid PEP 508 requirement"):
        render_pixi_manifest(metadata, ("linux-64",))


def test_platforms_for():
    assert platforms_for(("linux/amd64",)) == ("linux-64",)
    assert platforms_for(("linux/amd64", "linux/arm64")) == ("linux-64", "linux-aarch64")
    with pytest.raises(ValueError, match="Cannot build a pixi environment for platform 'windows/amd64'"):
        platforms_for(("windows/amd64",))


def test_check_platforms_supported(tmp_path: pathlib.Path):
    declared = (
        '# /// script\n# dependencies = ["cowsay"]\n#\n# [tool.pixi.workspace]\n# platforms = ["linux-64"]\n# ///\n'
    )
    metadata = parse_pixi_script_file(write(tmp_path, declared, name="declared.py"))

    # Covered: no complaint.
    check_platforms_supported(metadata, ("linux-64",))

    # Not covered: pixi would fail partway through the build, so say so up front.
    with pytest.raises(ValueError, match=r"do not cover \['linux-aarch64'\]"):
        check_platforms_supported(metadata, ("linux-64", "linux-aarch64"))

    # A script that declares nothing accepts whatever the image is built for.
    bare = parse_pixi_script_file(write(tmp_path, "# /// script\n# ///\n", name="silent.py"))
    check_platforms_supported(bare, ("linux-64", "linux-aarch64"))


def test_pixi_script_layer_validate_platforms(tmp_path: pathlib.Path):
    """A platform mismatch is caught by Image.validate(), before any build starts."""
    script = write(
        tmp_path,
        '# /// script\n# [tool.pixi.workspace]\n# platforms = ["linux-64"]\n# ///\n',
        name="mismatch.py",
    )
    img = Image.from_pixi_script(script, name="mismatch", registry="localhost")
    assert img._layers[-1].platforms == ("linux-64", "linux-aarch64")
    with pytest.raises(ValueError, match=r"do not cover \['linux-aarch64'\]"):
        img.validate()

    # Restricting the image to the declared platform resolves it.
    Image.from_pixi_script(script, name="mismatch", registry="localhost", platform=("linux/amd64",)).validate()


def test_pixi_script_layer_validate_script(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        PixiScript(script=tmp_path / "missing.py").validate()

    with pytest.raises(ValueError, match="is not a file"):
        PixiScript(script=tmp_path).validate()

    not_python = write(tmp_path, "# /// script\n# ///\n", name="script.txt")
    with pytest.raises(ValueError, match=r"must have a \.py extension"):
        PixiScript(script=not_python).validate()


def test_pixi_script_lock_discovery(tmp_path: pathlib.Path):
    """`pixi lock --script foo.py` writes foo.py.pixi.lock next to the script."""
    script = write(tmp_path, FULL_SCRIPT, name="locked.py")
    layer = PixiScript(script=script)
    assert layer.pixi_lock is None

    lock = tmp_path / "locked.py.pixi.lock"
    lock.write_text("version: 7")
    assert PixiScript(script=script).pixi_lock == lock
