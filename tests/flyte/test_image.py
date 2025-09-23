import sys
from pathlib import Path
from typing import cast

import pytest

from flyte._image import Image, PythonWheels, UVScript
from flyte._internal.imagebuild.docker_builder import PipAndRequirementsHandler


def test_base():
    img = Image.from_debian_base(name="test-image", registry="localhost:30000")
    assert img.name == "test-image"
    assert img.platform == ("linux/amd64", "linux/arm64")
    assert img.identifier


@pytest.mark.asyncio
async def test_with_requirements(tmp_path):
    file = tmp_path / "requirements.txt"
    with pytest.raises(FileNotFoundError):
        img = Image.from_debian_base(registry="localhost", name="test-image").with_requirements(file)
        await PipAndRequirementsHandler.handle(img._layers[-1], "/tmp", "")

    file = Path(__file__).parent / "resources" / "sample_requirements.txt"
    img = (
        Image.from_debian_base(registry="localhost", name="test-image")
        .with_requirements(file)
        .with_source_folder(Path("."))
    )
    assert img.identifier
    assert img._layers[-2].file == file


def test_with_pip_packages():
    packages = ("numpy", "pandas")
    img = Image.from_debian_base(registry="localhost", name="test-image").with_pip_packages(*packages)
    assert img._layers[-1].packages == packages

    img = Image.from_debian_base(registry="localhost", name="test-image").with_pip_packages(packages[0])
    assert img._layers[-1].packages == (packages[0],)

    img = Image.from_debian_base(registry="localhost", name="test-image").with_pip_packages(
        packages, extra_index_urls="https://example.com"
    )
    assert img._layers[-1].extra_index_urls == ("https://example.com",)


def test_with_source(tmp_path):
    file = tmp_path / "my_code.py"
    img = Image.from_debian_base(registry="localhost", name="test-image", flyte_version="0.2.0b14").with_source_file(
        file
    )
    assert img._layers[-1].src == file
    with pytest.raises(ValueError):
        img.validate()
    file.touch()
    img.validate()


def test_with_apt_packages():
    packages = ("curl", "vim")
    img = Image.from_debian_base(registry="localhost", name="test-image").with_apt_packages(*packages)
    assert img._layers[-1].packages == packages

    img = Image.from_debian_base(registry="localhost", name="test-image").with_apt_packages("curl")
    assert img._layers[-1].packages == ("curl",)


def test_with_workdir():
    workdir = "/app"
    img = Image.from_debian_base(registry="localhost", name="test-image").with_workdir(workdir)
    assert img._layers[-1].workdir == workdir


def test_default_base_image():
    default_image = Image.from_debian_base(flyte_version="2.0.0")
    assert default_image.uri.startswith("ghcr.io/flyteorg/flyte:py3.")
    assert default_image.identifier == "auto"
    default_image = Image.from_debian_base(python_version="3.12")
    assert not default_image.uri.startswith("ghcr.io/flyteorg/flyte:py3.")


def test_image_from_uv_script():
    script_path = Path(__file__).parent / "resources" / "sample_uv_script.py"
    img = Image.from_uv_script(script_path, name="uvtest", registry="localhost", python_version=(3, 12))
    assert img.uri.startswith("localhost/uvtest:")
    assert img._layers
    print(img._layers)
    assert isinstance(img._layers[-2], PythonWheels)
    assert isinstance(img._layers[-1], UVScript)
    script: UVScript = cast(UVScript, img._layers[-1])
    assert script.script == script_path
    assert img.uri.startswith("localhost/uvtest:")


def test_image_no_direct():
    with pytest.raises(TypeError):
        Image(base_image="python:3.13", name="test-image", registry="localhost:30000")


def test_raw_base_image():
    raw_base_image = Image.from_base("myregistry.com/myimage:v123")
    assert raw_base_image.uri == "myregistry.com/myimage:v123"
    if sys.version_info == (3, 13):
        assert raw_base_image.identifier == "yc2YEFXpndI_U6SJ5OnDbA"


def test_base_image_with_layers_unnamed():
    with pytest.raises(ValueError):
        Image.from_base("myregistry.com/myimage:v123").with_apt_packages("vim")


def test_base_image_with_layers():
    raw_base_image = (
        Image.from_base("myregistry.com/myimage:v123")
        .clone(registry="other_registry", name="myclone")
        .with_apt_packages("vim")
    )
    assert raw_base_image.uri == "other_registry/myclone:a95ad60ad5a34dd40c304b81cf9a15ae"
    assert len(raw_base_image._layers) == 1
    if sys.version_info == (3, 13):
        assert raw_base_image.identifier == "efpT8bNvJE39a-RbRJc78A"


def test_base_image_cloned():
    cloned_default_image = Image.from_debian_base(python_version=(3, 13)).clone(
        registry="ghcr.io/flyteorg", name="flyte-clone"
    )
    assert cloned_default_image.uri.startswith("ghcr.io/flyteorg/flyte-clone")
    if sys.version_info == (3, 13):
        assert cloned_default_image.identifier == "c7pJj0grD-FFWFPfHyUgRw"


def test_base_image_clone_same():
    default_image = Image.from_debian_base(python_version=(3, 13))
    cloned_default_image = Image.from_debian_base(python_version=(3, 13)).clone(
        registry="ghcr.io/flyteorg", name="random"
    )
    # These should not be the same because once cloned, the image loses its special tag
    assert default_image.uri != cloned_default_image.uri


def test_dockerfile():
    img = Image.from_dockerfile(
        file=Path(__file__).parent / "resources" / "Dockerfile.test_sample", name="test-image", registry="localhost"
    )
    assert img.uri.startswith("localhost/test-image")
    assert img.platform == ("linux/amd64",)
    img_multi = Image.from_dockerfile(
        file=Path(__file__).parent / "resources" / "Dockerfile.test_sample",
        name="test-image",
        registry="localhost",
        platform=("linux/amd64", "linux/arm64"),
    )
    assert img_multi.platform == ("linux/amd64", "linux/arm64")


def test_image_uri_consistency_for_uvscript():
    img = Image.from_uv_script(
        "./agent_simulation_loadtest.py", name="flyte", registry="ghcr.io/flyteorg", python_version=(3, 12)
    )
    assert img.base_image == "python:3.12-slim-bookworm", "Base image should be python:3.12-slim-bookworm"
    # This value should work across python versions in CI because all values have been specified above and are hardcoded
    # Please don't change this value unless you are sure it's the right thing to do.
    assert img.identifier == "ymxz6JlYRNMt5gqSOuEcSw", img._layers

def test_poetry_project_validate_missing_pyproject():
    import tempfile

    from flyte._image import PoetryProject

    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_pyproject = Path(tmpdir) / "non_existent_pyproject.toml"
        non_existent_poetry_lock = Path(tmpdir) / "non_existent_poetry.lock"
        poetry_project = PoetryProject(pyproject=non_existent_pyproject, poetry_lock=non_existent_poetry_lock)

        with pytest.raises(FileNotFoundError, match="pyproject.toml file .* does not exist"):
            poetry_project.validate()

def test_poetry_project_validate_missing_pyproject():
    import tempfile

    from flyte._image import PoetryProject

    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_pyproject = Path(tmpdir) / "non_existent_pyproject.toml"
        non_existent_poetry_lock = Path(tmpdir) / "non_existent_poetry.lock"
        poetry_project = PoetryProject(pyproject=non_existent_pyproject, poetry_lock=non_existent_poetry_lock)

        with pytest.raises(FileNotFoundError, match="pyproject.toml file .* does not exist"):
            poetry_project.validate()


def test_ids_for_different_python_version():
    ex_10 = Image.from_debian_base(python_version=(3, 10), install_flyte=False).with_source_file(Path(__file__))
    ex_11 = Image.from_debian_base(python_version=(3, 11), install_flyte=False).with_source_file(Path(__file__))
    ex_12 = Image.from_debian_base(python_version=(3, 12), install_flyte=False).with_source_file(Path(__file__))
    # Override base images to be the same for testing that the identifier does not depends on python version
    object.__setattr__(ex_11, "base_image", "python:3.10-slim-bookworm")
    object.__setattr__(ex_12, "base_image", "python:3.10-slim-bookworm")
    assert ex_10.identifier == ex_11.identifier

