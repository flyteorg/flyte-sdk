from pathlib import Path
from typing import cast

import pytest

from flyte._image import AptPackages, Image, UVScript
from flyte._internal.imagebuild.docker_builder import PipAndRequirementsHandler


def test_base():
    img = Image.from_debian_base(name="test-image", registry="localhost:30000")
    assert img.name == "test-image"
    assert img.platform == ("linux/amd64", "linux/arm64")


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

    # list of paths — each becomes its own layer
    file2 = tmp_path / "helper.py"
    file2.touch()
    img2 = Image.from_debian_base(registry="localhost", name="test-image", flyte_version="0.2.0b14").with_source_file(
        [file, file2]
    )
    assert img2._layers[-2].src == file
    assert img2._layers[-1].src == file2
    img2.validate()

    # duplicate filenames at the same dst must raise immediately
    subdir = tmp_path / "sub"
    subdir.mkdir()
    file3 = subdir / "my_code.py"  # same name as file
    with pytest.raises(ValueError, match="overwrite"):
        Image.from_debian_base(registry="localhost", name="test-image", flyte_version="0.2.0b14").with_source_file(
            [file, file3]
        )


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
    default_image = Image.from_debian_base(python_version="3.12")
    assert not default_image.uri.startswith("ghcr.io/flyteorg/flyte:py3.")


def test_image_from_uv_script():
    script_path = Path(__file__).parent / "resources" / "sample_uv_script.py"
    img = Image.from_uv_script(script_path, name="uvtest", registry="localhost", python_version=(3, 12))
    assert img.uri.startswith("localhost/uvtest:")
    assert img._layers
    print(img._layers)
    assert isinstance(img._layers[-2], AptPackages)
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


def test_base_image_with_layers_unnamed():
    with pytest.raises(ValueError):
        Image.from_base("myregistry.com/myimage:v123").with_apt_packages("vim")


def test_base_image_with_layers():
    raw_base_image = (
        Image.from_base("myregistry.com/myimage:v123")
        .clone(registry="other_registry", name="myclone", extendable=True)
        .with_apt_packages("vim")
    )
    assert raw_base_image.uri == "other_registry/myclone:a95ad60ad5a34dd40c304b81cf9a15ae"
    assert len(raw_base_image._layers) == 1


def test_base_image_cloned():
    cloned_default_image = Image.from_debian_base(python_version=(3, 13)).clone(
        registry="ghcr.io/flyteorg", name="flyte-clone"
    )
    assert cloned_default_image.uri.startswith("ghcr.io/flyteorg/flyte-clone")


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


def test_poetry_project_validate_missing_pyproject():
    import tempfile

    from flyte._image import PoetryProject

    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_pyproject = Path(tmpdir) / "non_existent_pyproject.toml"
        non_existent_poetry_lock = Path(tmpdir) / "non_existent_poetry.lock"
        poetry_project = PoetryProject(pyproject=non_existent_pyproject, poetry_lock=non_existent_poetry_lock)

        with pytest.raises(FileNotFoundError, match=r"pyproject.toml file .* does not exist"):
            poetry_project.validate()


def test_with_uv_project_optional_uvlock():
    """Test that with_uv_project correctly handles optional uvlock."""
    import tempfile

    from flyte._image import UVProject

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a pyproject.toml but no uv.lock file
        pyproject_file = Path(tmpdir) / "pyproject.toml"
        pyproject_file.write_text("[project]\nname = 'test-project'\nversion='0.1.0'")

        # Test that with_uv_project works without a uv.lock file
        img = Image.from_debian_base(registry="localhost", name="test-image").with_uv_project(
            pyproject_file=pyproject_file,
        )

        # Verify the layer is a UVProject with uvlock=None
        assert isinstance(img._layers[-1], UVProject)
        assert img._layers[-1].uvlock is None
        assert img._layers[-1].pyproject == pyproject_file

        # Now create a uv.lock file and verify it gets picked up
        uv_lock_file = Path(tmpdir) / "uv.lock"
        uv_lock_file.write_text("lock content")

        img2 = Image.from_debian_base(registry="localhost", name="test-image").with_uv_project(
            pyproject_file=pyproject_file,
        )
        # uvlock should be set to the default path since it now exists
        assert img2._layers[-1].uvlock == uv_lock_file


def test_uv_project_optional_uvlock():
    """Test that UVProject works correctly with optional uvlock."""
    import tempfile

    from flyte._image import UVProject

    with tempfile.TemporaryDirectory() as tmpdir:
        pyproject_file = Path(tmpdir) / "pyproject.toml"
        pyproject_file.write_text("[project]\nname = 'test-project'\nversion='0.1.0'")

        # Create UVProject without uvlock
        uv_project = UVProject(pyproject=pyproject_file, uvlock=None)
        assert uv_project.uvlock is None
        # Validate should pass even without uvlock
        uv_project.validate()

        # Test hash computation works without uvlock
        import hashlib

        hasher = hashlib.md5()
        uv_project.update_hash(hasher)
        hash1 = hasher.hexdigest()

        # Create with uvlock and verify hash is different
        uv_lock_file = Path(tmpdir) / "uv.lock"
        uv_lock_file.write_text("lock content")
        uv_project_with_lock = UVProject(pyproject=pyproject_file, uvlock=uv_lock_file)

        hasher2 = hashlib.md5()
        uv_project_with_lock.update_hash(hasher2)
        hash2 = hasher2.hexdigest()

        assert hash1 != hash2


def test_ids_for_different_python_version():
    ex_11 = Image.from_debian_base(python_version=(3, 11), install_flyte=False).with_source_file(Path(__file__))
    ex_12 = Image.from_debian_base(python_version=(3, 12), install_flyte=False).with_source_file(Path(__file__))
    # Override base images to be the same for testing that the identifier does not depends on python version
    object.__setattr__(ex_11, "base_image", "python:3.10-slim-bookworm")
    object.__setattr__(ex_12, "base_image", "python:3.10-slim-bookworm")


def test_layer_unhashable_type_error_message():
    """Test that Layer subclasses provide helpful error messages when lists are used instead of tuples."""
    from flyte._image import AptPackages, Commands, PipPackages

    # Test PipPackages with a list inside a tuple (common mistake)
    with pytest.raises(TypeError) as exc_info:
        PipPackages(packages=(["numpy", "pandas"],))  # tuple containing a list
    assert "packages" in str(exc_info.value)
    assert "contains a list" in str(exc_info.value)
    assert "Hint" in str(exc_info.value)

    # Test AptPackages with a list instead of tuple
    with pytest.raises(TypeError) as exc_info:
        AptPackages(packages=["vim", "curl"])  # list instead of tuple
    assert "packages" in str(exc_info.value)
    assert "is a list" in str(exc_info.value)
    assert "Pass items as separate arguments" in str(exc_info.value)

    # Test Commands with a list instead of tuple
    with pytest.raises(TypeError) as exc_info:
        Commands(commands=["echo hello", "ls -la"])  # list instead of tuple
    assert "commands" in str(exc_info.value)
    assert "is a list" in str(exc_info.value)

    # Verify valid usage works
    valid_pip = PipPackages(packages=("numpy", "pandas"))
    assert valid_pip.packages == ("numpy", "pandas")

    valid_apt = AptPackages(packages=("vim", "curl"))
    assert valid_apt.packages == ("vim", "curl")


def test_extendable_default():
    """Test that from_debian_base creates extendable images by default."""
    img = Image.from_debian_base(registry="localhost", name="test-image")
    assert img.extendable is True


def test_extendable_true():
    """Test that images can be marked as extendable."""
    img = Image.from_debian_base(registry="localhost", name="test-image").clone(extendable=True)
    assert img.extendable is True


def test_extendable_defaults_to_false_on_clone():
    """Test that cloning defaults to extendable=False."""
    # Start with extendable=True (from_debian_base default)
    img1 = Image.from_debian_base(registry="localhost", name="test-image")
    assert img1.extendable is True

    # Clone without specifying extendable - should default to False
    img2 = img1.clone(name="cloned-image")
    assert img2.extendable is True

    # Clone with extendable=True to keep it extendable
    img3 = img1.clone(name="still-extendable", extendable=True)
    assert img3.extendable is True

    # Clone without specifying extendable - should default to False (not preserve True)
    img4 = img3.clone(name="another-clone", extendable=False)
    assert img4.extendable is False

    # Must explicitly set extendable=True to keep it extendable
    img5 = img3.clone(name="explicitly-extendable", extendable=True)
    assert img5.extendable is True


def test_extendable_can_change_on_clone():
    """Test that extendable value can be explicitly changed when cloning."""
    img1 = Image.from_debian_base(registry="localhost", name="test-image")
    assert img1.extendable is True

    # Explicitly make it non-extendable
    img2 = img1.clone(name="non-extendable", extendable=False)
    assert img2.extendable is False

    # Make it extendable again
    img3 = img2.clone(name="extendable-again", extendable=True)
    assert img3.extendable is True


def test_extendable_allows_layers():
    """Test that extendable images can have layers added."""
    img = Image.from_debian_base(registry="localhost", name="test-image")
    assert img.extendable is True

    # Should not raise any errors
    img2 = img.with_pip_packages("numpy", "pandas")
    assert len(img2._layers) > len(img._layers)

    img3 = img2.with_apt_packages("vim")
    assert len(img3._layers) > len(img2._layers)


def test_from_debian_base_is_extendable_by_default():
    """Test that images created with from_debian_base are extendable by default."""
    img = Image.from_debian_base(registry="localhost", name="test-image")
    assert img.extendable is True

    # Should be able to add layers
    img2 = img.with_pip_packages("numpy")
    assert len(img2._layers) > len(img._layers)


def test_from_dockerfile_is_not_extendable():
    """Test that images created with from_dockerfile are not extendable by default."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".Dockerfile", delete=False) as f:
        f.write("FROM python:3.12-slim\n")
        f.write("RUN echo 'test'\n")
        dockerfile_path = Path(f.name)

    try:
        img = Image.from_dockerfile(file=dockerfile_path, registry="localhost", name="test-image")
        assert img.extendable is False
    finally:
        dockerfile_path.unlink()
