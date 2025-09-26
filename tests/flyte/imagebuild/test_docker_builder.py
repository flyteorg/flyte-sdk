import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from flyte import Secret
from flyte._image import Image, PipPackages, PoetryProject, Requirements
from flyte._internal.imagebuild.docker_builder import (
    CopyConfig,
    CopyConfigHandler,
    DockerImageBuilder,
    PipAndRequirementsHandler,
    PoetryProjectHandler,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basic_image():
    img = Image.from_debian_base(registry="localhost:30000", name="test_image", install_flyte=False).with_pip_packages(
        "requests"
    )

    builder = DockerImageBuilder()

    await builder.build_image(img, dry_run=False)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_image_folders_commands():
    img = (
        Image.from_debian_base(registry="localhost:30000", name="img_with_more", install_flyte=False)
        .with_pip_packages("requests")
        .with_source_folder(Path("."), "/root/data/stuff")
        .with_commands(["echo hello world", "echo hello world again"])
    )

    builder = DockerImageBuilder()
    await builder.build_image(img, dry_run=False)


@pytest.mark.skip("TemporaryDirectory.__init__() got an unexpected keyword argument 'delete")
@pytest.mark.asyncio
async def test_doesnt_work_yet():
    default_image = Image.from_debian_base()
    builder = DockerImageBuilder()
    await builder.build_image(default_image, dry_run=False)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_image_with_secrets(monkeypatch):
    monkeypatch.setenv("FLYTE", "test-value")
    monkeypatch.setenv("GROUP_KEY", "test-value")

    img = (
        Image.from_debian_base(registry="localhost:30000", name="img_with_secrets")
        .with_apt_packages("vim", secret_mounts="flyte")
        .with_pip_packages("requests", secret_mounts=[Secret(group="group", key="key")])
        .with_commands(["echo foobar"], secret_mounts=[Secret(group="group", key="key")])
    )

    builder = DockerImageBuilder()
    await builder.build_image(img)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("secret_mounts", [["flyte"], [Secret(group="group", key="key")]])
async def test_image_with_secrets_fails_if_secret_missing(secret_mounts):
    base = Image.from_debian_base(registry="localhost:30000", name="img_with_missing_secrets")
    builder = DockerImageBuilder()

    for func in [
        lambda img: img.with_apt_packages("vim", secret_mounts=secret_mounts),
        lambda img: img.with_pip_packages("requests", secret_mounts=secret_mounts),
        lambda img: img.with_commands(["echo foobar"], secret_mounts=secret_mounts),
    ]:
        layered = func(base)
        with pytest.raises(FileNotFoundError, match="Secret not found"):
            await builder.build_image(layered)


@pytest.mark.asyncio
async def test_pip_package_handling(monkeypatch):
    secret_mounts = (Secret("my-secret"), Secret("my-secret2"))
    monkeypatch.setenv("MY_SECRET", "test-value")

    # Create a temporary directory to simulate the context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_path = Path(tmpdir)

        # raw pip packages
        pip_packages = PipPackages(packages=("pkg_a", "pkg_b"), secret_mounts=secret_mounts)
        docker_update = await PipAndRequirementsHandler.handle(
            layer=pip_packages, context_path=context_path, dockerfile=""
        )
        assert "--mount=type=secret" in docker_update
        assert "uv pip install --python $UV_PYTHON pkg_a pkg_b" in docker_update


@pytest.mark.asyncio
async def test_requirements_handler(monkeypatch):
    secret_mounts = (Secret("my-secret"), Secret("my-secret2"))
    monkeypatch.setenv("MY_SECRET", "test-value")

    # Create a temporary directory to simulate the context
    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        with tempfile.TemporaryDirectory() as tmp_user_folder:
            user_folder = Path(tmp_user_folder)
            # create a dummy requirements.txt file
            requirements_file = user_folder / "requirements.txt"
            requirements_file.write_text("pkg_a\npkg_b\n")

            requirements = Requirements(file=requirements_file.absolute(), secret_mounts=secret_mounts)
            docker_update = await PipAndRequirementsHandler.handle(
                layer=requirements, context_path=context_path, dockerfile=""
            )
            assert "--mount=type=secret" in docker_update
            assert "_flyte_abs_context" + str(requirements_file.absolute()) in docker_update


@pytest.mark.asyncio
async def test_copy_config_handler():
    """Test handle method happy path - file exists and gets copied successfully"""
    # Create a temporary directory for context
    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        # Create a temporary file that will be copied
        with tempfile.TemporaryDirectory() as tmp_src_dir:
            src_dir = Path(tmp_src_dir)
            test_file = src_dir / "main.py"
            test_file.write_text("print('hello')")

            # Create CopyConfig for the file
            copy_config = CopyConfig(
                src=test_file,
                dst="/app/main.py",
                path_type=0,  # file
                src_name=test_file.name,
            )

            # Test the handle method
            result = await CopyConfigHandler.handle(
                layer=copy_config,
                context_path=context_path,
                dockerfile="FROM python:3.9\n",
                docker_ignore_file_path=None,
            )

            # Should contain COPY command when file is copied
            assert "COPY" in result
            assert "main.py" in result
            assert "/app/main.py" in result
            # Should return dockerfile with COPY command added
            assert result != "FROM python:3.9\n"

            # Verify that the file was actually copied to the correct destination path
            src_absolute = test_file.absolute()
            dst_path_str = str(src_absolute).replace("/", "./_flyte_abs_context/", 1)
            expected_dst_path = context_path / dst_path_str

            # Verify that the file was actually copied to the expected destination
            assert expected_dst_path.exists(), f"File should be copied to {expected_dst_path}"
            assert expected_dst_path.read_text() == "print('hello')", "File content should match"


@pytest.mark.asyncio
async def test_copy_config_handler_skips_dockerignore():
    """Test that handle method skips copying file when it matches various dockerignore patterns"""
    # Create a temporary directory for context
    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        # Create a temporary directory structure with both file and folder patterns
        with tempfile.TemporaryDirectory() as src_tmpdir:
            from flyte._internal.imagebuild.docker_builder import CopyConfig

            src_dir = Path(src_tmpdir)

            # Create nested directory structure: src_dir/src/utils/
            cache_dir = src_dir / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "temp.txt"
            cache_file.write_text("temp")

            # Create files in different locations
            root_file = src_dir / "main.py"
            root_file.write_text("print('hello from root')")
            exclude_file = src_dir / "memo.txt"
            exclude_file.write_text("memo")

            # Create .dockerignore file in the root directory (not in src_dir)
            dockerignore_file = src_dir / ".dockerignore"
            dockerignore_file.write_text("*.txt\n.cache\n")

            # Mock _get_init_config().root_dir to return src_dir
            with patch("flyte._initialize._get_init_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.root_dir = src_dir

                # Test copying the entire directory (path_type=1)
                copy_config = CopyConfig(
                    src=src_dir,
                    dst=".",
                    path_type=1,  # directory
                    src_name=src_dir.name,
                )

                result = await CopyConfigHandler.handle(
                    layer=copy_config,
                    context_path=context_path,
                    dockerfile="FROM python:3.9\n",
                    docker_ignore_file_path=None,
                )

                # Should contain COPY command for the directory
                assert "COPY" in result

                # Calculate the expected destination path using the same logic as handle method
                src_absolute = src_dir.absolute()
                dst_path_str = str(src_absolute).replace("/", "./_flyte_abs_context/", 1)
                expected_dst_path = context_path / dst_path_str

                # Verify that the directory was copied and ignored files are excluded
                assert expected_dst_path.exists(), f"Directory should be copied to {expected_dst_path}"
                assert expected_dst_path.is_dir(), "Should be a directory"
                assert (expected_dst_path / "main.py").exists(), "main.py should be included"
                assert not (expected_dst_path / "memo.txt").exists(), "memo.txt should be excluded"
                assert not (expected_dst_path / ".cache").exists(), ".cache directory should be excluded"


@pytest.mark.asyncio
async def test_copy_config_handler_with_dockerignore_layer():
    """Test CopyConfigHandler.handle respects DockerIgnore layer patterns"""
    # Create separate temporary directories for source and context
    with tempfile.TemporaryDirectory() as src_tmpdir:
        with tempfile.TemporaryDirectory() as context_tmpdir:
            src_dir = Path(src_tmpdir)
            context_path = Path(context_tmpdir)

            # Create test directory structure
            cache_dir = src_dir / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "temp.txt"
            cache_file.write_text("temp")

            root_file = src_dir / "main.py"
            root_file.write_text("print('hello from root')")
            exclude_file = src_dir / "memo.txt"
            exclude_file.write_text("memo")

            # Create .dockerignore file for DockerIgnore layer
            layer_dockerignore = src_dir / "custom.dockerignore"
            layer_dockerignore.write_text("*.txt\n.cache\n")

            # Mock _get_init_config().root_dir to return src_dir
            with patch("flyte._initialize._get_init_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.root_dir = src_dir

                # Create CopyConfig with DockerIgnore layer
                copy_config = CopyConfig(
                    src=src_dir,
                    dst=".",
                    path_type=1,  # directory
                    src_name=src_dir.name,
                )

                result = await CopyConfigHandler.handle(
                    layer=copy_config,
                    context_path=context_path,
                    dockerfile="FROM python:3.9\n",
                    docker_ignore_file_path=layer_dockerignore,
                )

                # Verify COPY command exists
                assert "COPY" in result

                # Calculate expected destination path
                src_absolute = src_dir.absolute()
                dst_path_str = str(src_absolute).replace("/", "./_flyte_abs_context/", 1)
                expected_dst_path = context_path / dst_path_str

                # Verify directory copy results and file exclusions
                assert expected_dst_path.exists(), f"Directory should be copied to {expected_dst_path}"
                assert expected_dst_path.is_dir(), "Should be a directory"
                assert (expected_dst_path / "main.py").exists(), "main.py should be included"
                assert not (expected_dst_path / "memo.txt").exists(), "memo.txt should be excluded"
                assert not (expected_dst_path / ".cache").exists(), ".cache directory should be excluded"


def test_list_dockerignore_layer_priority():
    """Test list_dockerignore method prioritizes DockerIgnore layer over local .dockerignore"""
    with tempfile.TemporaryDirectory() as tmp_context:
        src_dir = Path(tmp_context)

        # Create local .dockerignore file
        local_dockerignore = src_dir / ".dockerignore"
        local_dockerignore.write_text("*.py\nsrc/\n")

        # Create another .dockerignore file to be used as DockerIgnore layer
        layer_dockerignore = src_dir / "custom.dockerignore"
        layer_dockerignore.write_text("*.txt\n.cache\n")

        # Test list_dockerignore method with DockerIgnore layer path
        patterns = CopyConfigHandler.list_dockerignore(src_dir, layer_dockerignore)

        # Should return patterns from DockerIgnore layer instead of local .dockerignore
        expected_patterns = ["*.txt", ".cache"]
        assert patterns == expected_patterns


def test_list_dockerignore_found():
    """Test list_dockerignore method when .dockerignore file exists"""
    with tempfile.TemporaryDirectory() as tmp_context:
        src_dir = Path(tmp_context)

        # Create .dockerignore file with various patterns
        dockerignore_file = src_dir / ".dockerignore"
        dockerignore_file.write_text("*.py\nsrc/\n.cache\n# This is a comment\n\n*.txt\n \n  \n\t\n")

        # Test the method
        patterns = CopyConfigHandler.list_dockerignore(src_dir, None)

        # Should return expected patterns, excluding comments and empty lines
        expected_patterns = ["*.py", "src/", ".cache", "*.txt"]
        assert patterns == expected_patterns


def test_list_dockerignore_not_found():
    """Test list_dockerignore method when .dockerignore file does not exist"""
    with tempfile.TemporaryDirectory() as tmp_context:
        src_dir = Path(tmp_context)

        # Test the method
        patterns = CopyConfigHandler.list_dockerignore(src_dir, None)

        # Should return empty list when .dockerignore doesn't exist
        assert patterns == []


@pytest.mark.asyncio
async def test_poetry_handler_without_project_install():
    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        with tempfile.TemporaryDirectory() as tmp_user_folder:
            user_folder = Path(tmp_user_folder)
            pyproject_file = user_folder / "pyproject.toml"
            pyproject_file.write_text("[tool.poetry]\nname = 'test-project'")

            poetry_lock_file = user_folder / "poetry.lock"
            poetry_lock_file.write_text("[[package]]\nname = 'requests'\nversion = '2.28.0'")

            poetry_project = PoetryProject(
                pyproject=pyproject_file.absolute(),
                poetry_lock=poetry_lock_file.absolute(),
                extra_args="--no-root",
                secret_mounts=None,
            )

            initial_dockerfile = "FROM python:3.9\n"
            result = await PoetryProjectHandler.handel(poetry_project, context_path, initial_dockerfile)

            assert "RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv" in result
            assert "RUN --mount=type=cache,sharing=locked,mode=0777,target=/tmp/poetry_cache,id=poetry" in result
            assert "--mount=type=bind,target=poetry.lock,src=" in result
            assert "--mount=type=bind,target=pyproject.toml,src=" in result
            assert "uv pip install poetry" in result
            assert "ENV POETRY_CACHE_DIR=/tmp/poetry_cache" in result
            assert "POETRY_VIRTUALENVS_IN_PROJECT=true" in result
            assert "poetry install --no-root" in result


@pytest.mark.asyncio
async def test_poetry_handler_with_project_install():
    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        with tempfile.TemporaryDirectory() as tmp_user_folder:
            user_folder = Path(tmp_user_folder)
            pyproject_file = user_folder / "pyproject.toml"
            pyproject_file.write_text("[tool.poetry]\nname = 'test-project'")
            poetry_lock_file = user_folder / "poetry.lock"
            poetry_lock_file.write_text("[[package]]\nname = 'requests'\nversion = '2.28.0'")

            # Create PoetryProject without --no-root flag
            poetry_project = PoetryProject(pyproject=pyproject_file.absolute(), poetry_lock=None)

            initial_dockerfile = "FROM python:3.9\n"
            result = await PoetryProjectHandler.handel(poetry_project, context_path, initial_dockerfile)

            assert result.startswith(initial_dockerfile)

            assert "RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv" in result
            assert "RUN --mount=type=cache,sharing=locked,mode=0777,target=/tmp/poetry_cache,id=poetry" in result
            assert "uv pip install poetry" in result
            assert "ENV POETRY_CACHE_DIR=/tmp/poetry_cache" in result
            assert "POETRY_VIRTUALENVS_IN_PROJECT=true" in result
