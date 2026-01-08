from pathlib import Path
from typing import cast

import pytest

from flyte._image import Image, UVScript
from flyte._internal.imagebuild.docker_builder import PipAndRequirementsHandler
from flyte._internal.imagebuild.image_builder import ImageBuildEngine


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
        *packages, extra_index_urls="https://example.com"
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
    default_image = Image.from_debian_base(python_version="3.12")
    assert not default_image.uri.startswith("ghcr.io/flyteorg/flyte:py3.")


def test_image_from_uv_script():
    script_path = Path(__file__).parent / "resources" / "sample_uv_script.py"
    img = Image.from_uv_script(script_path, name="uvtest", registry="localhost", python_version=(3, 12))
    assert img.uri.startswith("localhost/uvtest:")
    assert img._layers
    print(img._layers)
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
        .clone(registry="other_registry", name="myclone")
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
        "examples/genai/agent_simulation_loadtest.py", name="flyte", registry="ghcr.io/flyteorg", python_version=(3, 12)
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


def test_ids_for_different_python_version():
    ex_11 = Image.from_debian_base(python_version=(3, 11), install_flyte=False).with_source_file(Path(__file__))
    ex_12 = Image.from_debian_base(python_version=(3, 12), install_flyte=False).with_source_file(Path(__file__))
    # Override base images to be the same for testing that the identifier does not depends on python version
    object.__setattr__(ex_11, "base_image", "python:3.10-slim-bookworm")
    object.__setattr__(ex_12, "base_image", "python:3.10-slim-bookworm")


def test_optimize_image_layers_single_layer():
    """Test optimization extracts heavy packages to a separate layer at the top."""
    from flyte._image import PipPackages

    img = Image.from_debian_base(registry="localhost", name="test-image", install_flyte=False).with_pip_packages(
        "torch", "tensorflow", "requests", "flask"
    )

    optimized = ImageBuildEngine._optimize_image_layers(img)
    pip_layers = [layer for layer in optimized._layers if isinstance(layer, PipPackages)]

    assert len(pip_layers) == 2
    # Heavy packages at top (torch and tensorflow)
    assert "torch" in pip_layers[0].packages
    assert "tensorflow" in pip_layers[0].packages
    # Light packages below (requests and flask)
    assert "requests" in pip_layers[1].packages
    assert "flask" in pip_layers[1].packages


def test_optimize_image_layers_multiple_layers():
    """Test optimization with multiple pip layers."""
    from flyte._image import PipPackages

    img = (
        Image.from_debian_base(registry="localhost", name="test-image", install_flyte=False)
        .with_pip_packages("torch", "requests")
        .with_pip_packages("tensorflow", "flask")
    )

    optimized = ImageBuildEngine._optimize_image_layers(img)
    pip_layers = [layer for layer in optimized._layers if isinstance(layer, PipPackages)]

    # Should have 4 layers:  2 heavy at top (torch, tensorflow), 2 light below (requests, flask)
    assert len(pip_layers) == 4

    # First two layers should be heavy packages
    heavy_packages = pip_layers[0].packages + pip_layers[1].packages
    assert "torch" in heavy_packages
    assert "tensorflow" in heavy_packages

    # Last two layers should be light packages
    light_packages = pip_layers[2].packages + pip_layers[3].packages
    assert "requests" in light_packages
    assert "flask" in light_packages


def test_optimize_image_layers_no_heavy_packages():
    """Test optimization when there are no heavy packages."""
    from flyte._image import PipPackages

    img = Image.from_debian_base(registry="localhost", name="test-image", install_flyte=False).with_pip_packages(
        "requests", "flask"
    )

    optimized = ImageBuildEngine._optimize_image_layers(img)

    # Should return the same image structure since no optimization needed
    original_pip_layers = [layer for layer in img._layers if isinstance(layer, PipPackages)]
    optimized_pip_layers = [layer for layer in optimized._layers if isinstance(layer, PipPackages)]
    assert len(optimized_pip_layers) == len(original_pip_layers)


def test_optimize_image_layers_only_heavy_packages():
    """Test optimization when a layer contains only heavy packages."""
    from flyte._image import PipPackages

    img = Image.from_debian_base(registry="localhost", name="test-image", install_flyte=False).with_pip_packages(
        "torch", "tensorflow"
    )

    optimized = ImageBuildEngine._optimize_image_layers(img)
    pip_layers = [layer for layer in optimized._layers if isinstance(layer, PipPackages)]

    # Should have 1 heavy layer at top (both torch and tensorflow are heavy)
    assert len(pip_layers) == 1
    assert "torch" in pip_layers[0].packages
    assert "tensorflow" in pip_layers[0].packages


def test_optimize_image_layers_preserves_extra_args():
    """Test that optimization preserves pip layer arguments like index_url."""
    from flyte._image import PipPackages

    img = (
        Image.from_debian_base(registry="localhost", name="test-image", install_flyte=False)
        .with_pip_packages("torch", "requests", extra_index_urls="https://example.com")
        .with_pip_packages("tensorflow", "flask", extra_index_urls="https://other.com")
    )

    optimized = ImageBuildEngine._optimize_image_layers(img)
    pip_layers = [layer for layer in optimized._layers if isinstance(layer, PipPackages)]

    # Find the heavy layer with torch
    torch_layer = pip_layers[0]
    assert torch_layer.extra_index_urls == ("https://example.com",)

    # Find the heavy layer with tensorflow
    tensorflow_layer = pip_layers[1]
    assert tensorflow_layer.extra_index_urls == ("https://other.com",)


def test_optimize_image_layers_with_non_pip_layers():
    """Test optimization preserves non-pip layers in correct positions."""
    from flyte._image import AptPackages, PipPackages

    img = (
        Image.from_debian_base(registry="localhost", name="test-image", install_flyte=False)
        .with_apt_packages("curl", "vim")
        .with_pip_packages("torch", "requests")
    )

    optimized = ImageBuildEngine._optimize_image_layers(img)

    # Apt layers should still exist
    apt_layers = [layer for layer in optimized._layers if isinstance(layer, AptPackages)]
    assert len(apt_layers) >= 1  # At least one apt layer (base + custom)

    # Should have pip layers
    pip_layers = [layer for layer in optimized._layers if isinstance(layer, PipPackages)]
    assert len(pip_layers) >= 1


def test_optimize_image_layers_flyte_wheels_at_end():
    """Test that PythonWheels with package_name 'flyte' are moved to the end."""
    from flyte._image import PythonWheels

    img = Image.from_debian_base(registry="localhost", name="test-image")

    # The default image should have flyte wheels
    optimized = ImageBuildEngine._optimize_image_layers(img)

    # Find flyte wheel layers
    flyte_wheels = [
        layer for layer in optimized._layers if isinstance(layer, PythonWheels) and layer.package_name == "flyte"
    ]

    if flyte_wheels:
        # Flyte wheels should be at the very end
        last_flyte_wheel_index = optimized._layers.index(flyte_wheels[-1])
        assert last_flyte_wheel_index == len(optimized._layers) - 1


def test_optimize_image_layers_with_uv_script():
    """Test optimization with UVScript layers."""
    from flyte._image import UVScript

    script_path = Path(__file__).parent / "resources" / "sample_uv_script.py"

    # Skip test if file doesn't exist
    if not script_path.exists():
        pytest.skip(f"Test file not found: {script_path}")

    img = Image.from_uv_script(script_path, name="uvtest", registry="localhost", python_version=(3, 12))

    optimized = ImageBuildEngine._optimize_image_layers(img)

    # UVScript layer should still exist
    uv_layers = [layer for layer in optimized._layers if isinstance(layer, UVScript)]
    assert len(uv_layers) >= 1
