from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from flyte._image import (
    _BASE_REGISTRY,
    _LOCALHOST_REGISTRY,
    AptPackages,
    CodeBundleLayer,
    CopyConfig,
    Image,
    UVScript,
    _get_base_registry,
    resolve_code_bundle_layer,
)
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


def test_with_requirements_index_url(tmp_path):
    file = Path(__file__).parent / "resources" / "sample_requirements.txt"
    img = Image.from_debian_base(registry="localhost", name="test-image").with_requirements(
        file, index_url="https://my-private-pypi.example.com/simple"
    )
    layer = img._layers[-1]
    assert layer.file == file
    assert layer.index_url == "https://my-private-pypi.example.com/simple"


def test_with_requirements_extra_index_urls(tmp_path):
    file = Path(__file__).parent / "resources" / "sample_requirements.txt"
    img = Image.from_debian_base(registry="localhost", name="test-image").with_requirements(
        file, extra_index_urls="https://extra.example.com/simple"
    )
    layer = img._layers[-1]
    assert layer.extra_index_urls == ("https://extra.example.com/simple",)

    img2 = Image.from_debian_base(registry="localhost", name="test-image").with_requirements(
        file, extra_index_urls=["https://extra1.example.com", "https://extra2.example.com"]
    )
    layer2 = img2._layers[-1]
    assert layer2.extra_index_urls == ("https://extra1.example.com", "https://extra2.example.com")


def test_with_requirements_pre(tmp_path):
    file = Path(__file__).parent / "resources" / "sample_requirements.txt"
    img = Image.from_debian_base(registry="localhost", name="test-image").with_requirements(file, pre=True)
    layer = img._layers[-1]
    assert layer.pre is True


def test_with_requirements_extra_args(tmp_path):
    file = Path(__file__).parent / "resources" / "sample_requirements.txt"
    img = Image.from_debian_base(registry="localhost", name="test-image").with_requirements(
        file, extra_args="--no-deps"
    )
    layer = img._layers[-1]
    assert layer.extra_args == "--no-deps"


def test_with_requirements_all_pip_options(tmp_path):
    file = Path(__file__).parent / "resources" / "sample_requirements.txt"
    img = Image.from_debian_base(registry="localhost", name="test-image").with_requirements(
        file,
        index_url="https://private.example.com/simple",
        extra_index_urls=["https://extra.example.com"],
        pre=True,
        extra_args="--no-deps",
    )
    layer = img._layers[-1]
    assert layer.file == file
    assert layer.index_url == "https://private.example.com/simple"
    assert layer.extra_index_urls == ("https://extra.example.com",)
    assert layer.pre is True
    assert layer.extra_args == "--no-deps"


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

    # Single-file COPY layers must hash file contents (os.walk on a file yields nothing).
    f = tmp_path / "version.md"
    f.write_text("first")
    base = Image.from_debian_base(registry="localhost", name="test-image", flyte_version="0.2.0b14")
    h1 = base.with_source_file(f)._get_hash_digest()
    f.write_text("second")
    h2 = base.with_source_file(f)._get_hash_digest()
    assert h1 != h2

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


def test_released_version_returns_prebuilt_image():
    """When a released flyte_version is provided, from_debian_base should return a
    prebuilt Image.from_base() reference with no build layers."""
    img = Image.from_debian_base(flyte_version="2.0.7")
    assert img.uri == "ghcr.io/flyteorg/flyte:py3.12-v2.0.7" or img.uri.startswith("ghcr.io/flyteorg/flyte:py3.")
    assert "v2.0.7" in img.uri


def test_released_version_with_v_prefix():
    """flyte_version starting with 'v' should not double-prefix."""
    img = Image.from_debian_base(flyte_version="v2.0.7")
    assert "v2.0.7" in img.uri
    assert "vv" not in img.uri


def test_install_flyte_false_builds_full_image():
    """When install_flyte=False, should build a full image (not prebuilt)."""
    img = Image.from_debian_base(install_flyte=False)
    # Should have layers since it's building a full image
    assert len(img._layers) > 0
    # Tag should not contain a version suffix
    assert img.uri.startswith("ghcr.io/flyteorg/flyte:py3.")


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


def test_default_image_creates_flyte_user():
    """The default debian-base image should add a Commands layer that creates the flyte user
    and a WorkDir layer that sets the working directory to /home/flyte. Both layers are
    proto-backed so they're picked up by the remote image builder as well."""
    from flyte._image import Commands, WorkDir

    img = Image.from_debian_base(install_flyte=False)
    layer_types = [type(layer) for layer in img._layers]

    # Should contain a Commands layer (user creation) and a WorkDir layer
    assert Commands in layer_types, f"Default image is missing a Commands layer. Got: {layer_types}"
    assert WorkDir in layer_types, f"Default image is missing a WorkDir layer. Got: {layer_types}"

    # Find the Commands layer for user creation and verify its contents
    commands_layers = [layer for layer in img._layers if isinstance(layer, Commands)]
    user_commands = [
        cmd for layer in commands_layers for cmd in layer.commands if "useradd" in cmd and "flyte" in cmd
    ]
    assert user_commands, f"Expected a Commands layer that creates the flyte user. Got: {commands_layers}"

    user_create_cmd = user_commands[0]
    # Idempotent user creation
    assert "id -u flyte" in user_create_cmd
    assert "useradd --create-home --shell /bin/bash flyte" in user_create_cmd
    # Should chown the home directory and /root so the runtime user can write there
    assert "chown -R flyte:flyte /home/flyte" in user_create_cmd
    assert "chown -R flyte:flyte /root" in user_create_cmd

    # Verify WorkDir is /home/flyte
    workdir_layers = [layer for layer in img._layers if isinstance(layer, WorkDir)]
    assert any(layer.workdir == "/home/flyte" for layer in workdir_layers), (
        f"Expected WorkDir(/home/flyte). Got: {workdir_layers}"
    )


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


def test_dockerfile_with_str_path():
    img = Image.from_dockerfile(
        file=str(Path(__file__).parent / "resources/Dockerfile.test_sample"),
        registry="localhost",
        name="test-image",
    )
    assert img.uri.startswith("localhost/test-image"), f"Unexpected URI: {img.uri}"
    assert img.platform == ("linux/amd64",)


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


def test_copy_config_update_hash_respects_dockerignore(tmp_path):
    """CopyConfig.update_hash must exclude files matched by .dockerignore."""
    import hashlib

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("hello")
    (src / "b.txt").write_text("world")
    (src / "c.txt").write_text("data")

    # Hash without any .dockerignore — all three files included
    h_all = hashlib.md5()
    CopyConfig(path_type=1, src=src, dst="/app").update_hash(h_all)
    digest_all = h_all.hexdigest()

    # Add a .dockerignore that excludes c.txt
    (src / ".dockerignore").write_text("c.txt\n")

    h_ignored = hashlib.md5()
    CopyConfig(path_type=1, src=src, dst="/app").update_hash(h_ignored)
    digest_ignored = h_ignored.hexdigest()

    # Hash must differ because c.txt is excluded (and .dockerignore itself is now included)
    assert digest_all != digest_ignored


def test_copy_config_dockerignore_itself_is_hashed(tmp_path):
    """Changing .dockerignore content must change the hash (PatternMatcher always includes it)."""
    import hashlib

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("hello")
    (src / ".dockerignore").write_text("*.log\n")

    h1 = hashlib.md5()
    CopyConfig(path_type=1, src=src, dst="/app").update_hash(h1)

    # Change .dockerignore content — hash must change because the file content changed
    (src / ".dockerignore").write_text("*.log\n*.tmp\n")
    h2 = hashlib.md5()
    CopyConfig(path_type=1, src=src, dst="/app").update_hash(h2)

    assert h1.hexdigest() != h2.hexdigest()


def test_uv_project_install_project_respects_dockerignore(tmp_path):
    """UVProject (install_project mode) must exclude files matched by .dockerignore."""
    import hashlib

    from flyte._image import UVProject

    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname='p'\nversion='0.1.0'")
    (project_dir / "main.py").write_text("print('hello')")
    (project_dir / "data.bin").write_text("big data")

    h_all = hashlib.md5()
    UVProject(pyproject=project_dir / "pyproject.toml", project_install_mode="install_project").update_hash(h_all)
    digest_all = h_all.hexdigest()

    # Add .dockerignore to exclude data.bin
    (project_dir / ".dockerignore").write_text("data.bin\n")

    h_ignored = hashlib.md5()
    UVProject(pyproject=project_dir / "pyproject.toml", project_install_mode="install_project").update_hash(h_ignored)

    assert digest_all != h_ignored.hexdigest()


def test_copy_config_respects_dockerignore_from_image_layer(tmp_path):
    """
    .dockerignore specified via with_dockerignore() at the project root must be
    respected by CopyConfig even when src is a subdirectory.

    The pattern is relative to the .dockerignore location (project root), so
    a file at project/src/data.bin is excluded by the pattern 'src/data.bin'.
    Both images share the same DockerIgnore layer so only the CopyConfig hash differs.
    """
    from flyte._image import Image

    project = tmp_path / "project"
    project.mkdir()
    src = project / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hello')")
    (src / "data.bin").write_text("big data")

    # .dockerignore is at project root, NOT inside src/.
    # Pattern is relative to the project root.
    di = project / ".dockerignore"
    di.write_text("src/data.bin\n")

    img = Image.from_debian_base(registry="localhost", name="test").with_dockerignore(di).with_source_folder(src)

    # Reference: same dockerignore layer + same dst name, but src has no data.bin.
    # Use the same directory name so with_source_folder produces the same dst.
    project2 = tmp_path / "project2"
    project2.mkdir()
    src_clean = project2 / "src"
    src_clean.mkdir()
    (src_clean / "main.py").write_text("print('hello')")

    img_clean = (
        Image.from_debian_base(registry="localhost", name="test").with_dockerignore(di).with_source_folder(src_clean)
    )

    assert img.uri == img_clean.uri


def test_dockerignore_hash_changes_with_content(tmp_path):
    """Changing .dockerignore contents must produce a different hash."""
    import hashlib

    from flyte._image import DockerIgnore

    di_file = tmp_path / ".dockerignore"
    di_file.write_text("*.log\n")
    h1 = hashlib.md5()
    DockerIgnore(path=str(di_file)).update_hash(h1)

    di_file.write_text("*.log\n*.pyc\n")
    h2 = hashlib.md5()
    DockerIgnore(path=str(di_file)).update_hash(h2)

    assert h1.hexdigest() != h2.hexdigest()


def test_dockerignore_hash_stable_for_same_content(tmp_path):
    """Same .dockerignore contents must produce a stable hash."""
    import hashlib

    from flyte._image import DockerIgnore

    di_file = tmp_path / ".dockerignore"
    di_file.write_text("*.log\n")
    layer = DockerIgnore(path=str(di_file))

    h1, h2 = hashlib.md5(), hashlib.md5()
    layer.update_hash(h1)
    layer.update_hash(h2)
    assert h1.hexdigest() == h2.hexdigest()


def test_image_uri_changes_when_dockerignore_content_changes(tmp_path):
    """Image URI (cache key) must differ when .dockerignore contents change."""
    di_file = tmp_path / ".dockerignore"
    di_file.write_text("*.log\n")
    img1 = Image.from_debian_base(registry="localhost", name="test").with_dockerignore(di_file)
    uri1 = img1.uri  # access before overwriting the file

    di_file.write_text("*.log\n*.pyc\n")
    img2 = Image.from_debian_base(registry="localhost", name="test").with_dockerignore(di_file)

    assert uri1 != img2.uri


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


def test_with_code_bundle_defaults():
    """with_code_bundle() creates a CodeBundleLayer with default values."""
    img = Image.from_debian_base(registry="localhost", name="test-image").with_code_bundle()
    layer = img._layers[-1]
    assert isinstance(layer, CodeBundleLayer)
    assert layer.copy_style == "loaded_modules"
    assert layer.dst == "."


def test_with_code_bundle_all():
    """with_code_bundle(copy_style='all') stores the correct style."""
    img = Image.from_debian_base(registry="localhost", name="test-image").with_code_bundle(copy_style="all")
    layer = img._layers[-1]
    assert isinstance(layer, CodeBundleLayer)
    assert layer.copy_style == "all"


def test_with_code_bundle_custom_dst():
    """with_code_bundle(dst='/app') stores the custom destination."""
    img = Image.from_debian_base(registry="localhost", name="test-image").with_code_bundle(dst="/app")
    layer = img._layers[-1]
    assert isinstance(layer, CodeBundleLayer)
    assert layer.dst == "/app"


def test_resolve_code_bundle_no_layers():
    """resolve_code_bundle_layer returns the same object when no CodeBundleLayer layers present."""
    img = Image.from_debian_base(registry="localhost", name="test-image")
    result = resolve_code_bundle_layer(img, "loaded_modules", Path("/tmp"))
    assert result is img


def test_resolve_code_bundle_strips_when_not_none():
    """resolve_code_bundle_layer strips CodeBundleLayer when copy_style != 'none'."""
    img = Image.from_debian_base(registry="localhost", name="test-image").with_code_bundle()
    result = resolve_code_bundle_layer(img, "loaded_modules", Path("/tmp"))
    # Should have no CodeBundleLayer layers
    assert not any(isinstance(layer, CodeBundleLayer) for layer in result._layers)


def test_resolve_code_bundle_strips_when_all():
    """resolve_code_bundle_layer strips CodeBundleLayer when copy_style is 'all'."""
    img = Image.from_debian_base(registry="localhost", name="test-image").with_code_bundle()
    result = resolve_code_bundle_layer(img, "all", Path("/tmp"))
    assert not any(isinstance(layer, CodeBundleLayer) for layer in result._layers)


def test_resolve_code_bundle_hash_stability():
    """Stripped image should have the same hash as image without with_code_bundle()."""
    base = Image.from_debian_base(registry="localhost", name="test-image")
    with_bundle = base.with_code_bundle()
    stripped = resolve_code_bundle_layer(with_bundle, "loaded_modules", Path("/tmp"))
    assert base.uri == stripped.uri


def test_resolve_code_bundle_all_copy_style_none(tmp_path):
    """resolve_code_bundle_layer replaces with CopyConfig for copy_style='all' when runner is 'none'."""
    # Create a source directory with some files
    src_dir = tmp_path / "project"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("print('hello')")

    img = Image.from_debian_base(registry="localhost", name="test-image").with_code_bundle(copy_style="all")
    result = resolve_code_bundle_layer(img, "none", src_dir)

    # The CodeBundleLayer should be replaced with a CopyConfig
    assert not any(isinstance(layer, CodeBundleLayer) for layer in result._layers)
    copy_layers = [layer for layer in result._layers if isinstance(layer, CopyConfig)]
    assert len(copy_layers) == 1
    assert copy_layers[0].path_type == 1
    assert copy_layers[0].src == src_dir
    assert copy_layers[0].dst == "."


def test_resolve_code_bundle_loaded_modules_copy_style_none(tmp_path):
    """resolve_code_bundle_layer resolves 'loaded_modules' with root_dir set when runner is 'none'."""
    img = Image.from_debian_base(registry="localhost", name="test-image").with_code_bundle(copy_style="loaded_modules")
    result = resolve_code_bundle_layer(img, "none", tmp_path)

    # The CodeBundleLayer should be kept but with root_dir set
    bundle_layers = [layer for layer in result._layers if isinstance(layer, CodeBundleLayer)]
    assert len(bundle_layers) == 1
    assert bundle_layers[0].root_dir == tmp_path
    assert bundle_layers[0].copy_style == "loaded_modules"
    assert bundle_layers[0].dst == "."


def test_get_base_registry_returns_default_when_not_initialized():
    """When flyte is not initialized, _get_base_registry returns the default registry."""
    with patch("flyte._initialize._get_init_config", return_value=None):
        assert _get_base_registry() == _BASE_REGISTRY


def test_get_base_registry_returns_default_when_no_client():
    """When init config has no client, _get_base_registry returns the default registry."""
    mock_config = MagicMock()
    mock_config.client = None
    with patch("flyte._initialize._get_init_config", return_value=mock_config):
        assert _get_base_registry() == _BASE_REGISTRY


def test_get_base_registry_returns_default_for_remote_endpoint():
    """When endpoint is a remote URL, _get_base_registry returns the default registry."""
    mock_config = MagicMock()
    mock_config.client.endpoint = "dns:///my-cluster.example.com"
    with patch("flyte._initialize._get_init_config", return_value=mock_config):
        assert _get_base_registry() == _BASE_REGISTRY


def test_get_base_registry_returns_localhost_for_localhost_endpoint():
    """When endpoint contains 'localhost', _get_base_registry returns the localhost registry."""
    mock_config = MagicMock()
    mock_config.client.endpoint = "localhost:8090"
    with patch("flyte._initialize._get_init_config", return_value=mock_config):
        assert _get_base_registry() == _LOCALHOST_REGISTRY


def test_get_base_registry_returns_localhost_for_localhost_in_url():
    """When endpoint contains 'localhost' as part of a URL, _get_base_registry returns the localhost registry."""
    mock_config = MagicMock()
    mock_config.client.endpoint = "dns:///localhost:30080"
    with patch("flyte._initialize._get_init_config", return_value=mock_config):
        assert _get_base_registry() == _LOCALHOST_REGISTRY


def test_get_base_registry_returns_default_for_empty_endpoint():
    """When endpoint is empty string, _get_base_registry returns the default registry."""
    mock_config = MagicMock()
    mock_config.client.endpoint = ""
    with patch("flyte._initialize._get_init_config", return_value=mock_config):
        assert _get_base_registry() == _BASE_REGISTRY


def test_released_default_image_is_not_cloned():
    """A released, unmodified default image should have _is_cloned=False so the SDK skips building it."""
    with patch("flyte._version.__version__", "1.2.3"):
        image = Image.from_debian_base(python_version=(3, 12))
    assert image._is_cloned is False


def test_dev_default_image_is_cloned():
    """A dev-version default image should have _is_cloned=True so the SDK builds it."""
    with patch("flyte._version.__version__", "1.2.3.dev0+abc"):
        image = Image.from_debian_base(python_version=(3, 12))
    assert image._is_cloned is True


def test_with_pip_packages_marks_image_as_cloned():
    """Adding pip packages to a default image should flip _is_cloned to True."""
    with patch("flyte._version.__version__", "1.2.3"):
        image = Image.from_debian_base(
            python_version=(3, 12),
            registry="my-registry.example.com",
            name="my-image",
        )
        assert image._is_cloned is True  # registry override already triggers clone
        modified = image.with_pip_packages("requests")
    assert modified._is_cloned is True


def test_clone_marks_image_as_cloned():
    """Explicitly calling clone() should produce an image with _is_cloned=True."""
    image = Image.from_base("ghcr.io/example/my-image:latest")
    assert image._is_cloned is False
    cloned = image.clone(name="my-image")
    assert cloned._is_cloned is True


def test_from_debian_base_with_registry_is_cloned():
    """Overriding the registry on from_debian_base produces a cloned image (needs build)."""
    with patch("flyte._version.__version__", "1.2.3"):
        image = Image.from_debian_base(python_version=(3, 12), registry="my-registry.example.com", name="my-image")
    assert image._is_cloned is True


def test_from_base_is_not_cloned():
    """Image.from_base points at an existing image URI and should not be rebuilt unless modified."""
    image = Image.from_base("my-registry.example.com/my-image:latest")
    assert image._is_cloned is False


def test_from_base_with_layers_is_cloned():
    """Adding layers to a from_base image must flip _is_cloned so the SDK builds the derived image."""
    image = Image.from_base("my-registry.example.com/my-image:latest")
    # from_base images don't have a name, so we need to clone first to add layers.
    cloned = image.clone(name="derived", extendable=True).with_pip_packages("requests")
    assert cloned._is_cloned is True


def test_from_ref_name_is_not_cloned():
    """Image.from_ref_name is a pointer to an externally configured image and should not be rebuilt."""
    image = Image.from_ref_name("my-ref")
    assert image._is_cloned is False
