import asyncio
import subprocess
import tempfile
from pathlib import Path, PurePath
from unittest.mock import patch

import pytest
import pytest_asyncio

from flyte import Secret
from flyte._image import AptPackages, Commands, Image, PipPackages, PoetryProject, Requirements, UVProject
from flyte._internal.imagebuild.docker_builder import (
    DOCKER_FILE_UV_BASE_TEMPLATE,
    CopyConfig,
    CopyConfigHandler,
    DockerImageBuilder,
    PipAndRequirementsHandler,
    PoetryProjectHandler,
    UVProjectHandler,
    _get_secret_commands,
)
from flyte._internal.imagebuild.remote_builder import _get_build_secrets_from_image


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
async def test_pip_package_handling_with_version_constraints():
    """Package specs containing shell metacharacters (<, >) must be quoted in the generated Dockerfile
    so that the shell does not interpret them as redirection operators."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_path = Path(tmpdir)

        pip_packages = PipPackages(packages=("apache-airflow<=3.0.0", "requests>=2.0,<3"))
        docker_update = await PipAndRequirementsHandler.handle(
            layer=pip_packages, context_path=context_path, dockerfile=""
        )
        # Each spec with a shell metacharacter must be single-quoted
        assert "'apache-airflow<=3.0.0'" in docker_update
        assert "'requests>=2.0,<3'" in docker_update


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
            )

            # Test the handle method
            result = await CopyConfigHandler.handle(
                layer=copy_config,
                context_path=context_path,
                dockerfile="FROM python:3.9\n",
                docker_ignore_patterns=[],
            )

            # Should contain COPY command when file is copied
            assert "COPY" in result
            assert "main.py" in result
            assert "/app/main.py" in result
            # Should return dockerfile with COPY command added
            assert result != "FROM python:3.9\n"

            # Verify that the file was actually copied to the correct destination path
            src_absolute = test_file.absolute()
            rel_path = PurePath(*src_absolute.parts[1:])
            expected_dst_path = context_path / "_flyte_abs_context" / rel_path

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

            # Mock _get_init_config().root_dir to return src_dir
            with patch("flyte._initialize._get_init_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.root_dir = src_dir

                # Test copying the entire directory (path_type=1)
                copy_config = CopyConfig(
                    src=src_dir,
                    dst=".",
                    path_type=1,  # directory
                )

                result = await CopyConfigHandler.handle(
                    layer=copy_config,
                    context_path=context_path,
                    dockerfile="FROM python:3.9\n",
                    docker_ignore_patterns=["*.txt", ".cache"],
                )

                # Should contain COPY command for the directory
                assert "COPY" in result

                # Calculate the expected destination path using the same logic as handle method
                src_absolute = src_dir.absolute()
                rel_path = PurePath(*src_absolute.parts[1:])
                expected_dst_path = context_path / "_flyte_abs_context" / rel_path

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

            # Mock _get_init_config().root_dir to return src_dir
            with patch("flyte._initialize._get_init_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.root_dir = src_dir

                # Create CopyConfig with DockerIgnore layer
                copy_config = CopyConfig(
                    src=src_dir,
                    dst=".",
                    path_type=1,  # directory
                )

                result = await CopyConfigHandler.handle(
                    layer=copy_config,
                    context_path=context_path,
                    dockerfile="FROM python:3.9\n",
                    docker_ignore_patterns=["*.txt", ".cache"],
                )

                # Verify COPY command exists
                assert "COPY" in result

                # Calculate expected destination path
                src_absolute = src_dir.absolute()
                rel_path = PurePath(*src_absolute.parts[1:])
                expected_dst_path = context_path / "_flyte_abs_context" / rel_path

                # Verify directory copy results and file exclusions
                assert expected_dst_path.exists(), f"Directory should be copied to {expected_dst_path}"
                assert expected_dst_path.is_dir(), "Should be a directory"
                assert (expected_dst_path / "main.py").exists(), "main.py should be included"
                assert not (expected_dst_path / "memo.txt").exists(), "memo.txt should be excluded"
                assert not (expected_dst_path / ".cache").exists(), ".cache directory should be excluded"


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
            result = await PoetryProjectHandler.handle(
                layer=poetry_project,
                context_path=context_path,
                dockerfile=initial_dockerfile,
                docker_ignore_patterns=[],
            )

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
            poetry_project = PoetryProject(pyproject=pyproject_file.absolute(), poetry_lock=poetry_lock_file)

            cache_dir = user_folder / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "temp.txt"
            cache_file.write_text("temp")
            exclude_file = user_folder / "memo.txt"
            exclude_file.write_text("memo")
            # Create a file that should be included
            (user_folder / "main.py").write_text("print('hello')")

            initial_dockerfile = "FROM python:3.9\n"
            result = await PoetryProjectHandler.handle(
                layer=poetry_project,
                context_path=context_path,
                dockerfile=initial_dockerfile,
                docker_ignore_patterns=["*.txt", ".cache", "pyproject.toml", "*.toml", "poetry.lock", "*.lock"],
            )

            assert result.startswith(initial_dockerfile)

            assert "RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv" in result
            assert "RUN --mount=type=cache,sharing=locked,mode=0777,target=/tmp/poetry_cache,id=poetry" in result
            assert "uv pip install poetry" in result
            assert "ENV POETRY_CACHE_DIR=/tmp/poetry_cache" in result
            assert "POETRY_VIRTUALENVS_IN_PROJECT=true" in result

            # Calculate expected destination path
            src_absolute = user_folder.absolute()
            rel_path = PurePath(*src_absolute.parts[1:])
            expected_dst_path = context_path / "_flyte_abs_context" / rel_path

            # Verify directory copy results and file exclusions
            assert expected_dst_path.exists(), f"Directory should be copied to {expected_dst_path}"
            assert expected_dst_path.is_dir(), "Should be a directory"
            assert (expected_dst_path / "pyproject.toml").exists(), "pyproject.toml should be included"
            assert (expected_dst_path / "poetry.lock").exists(), "poetry.lock should be included"
            assert not (expected_dst_path / "memo.txt").exists(), "memo.txt should be excluded"
            assert not (expected_dst_path / ".cache").exists(), ".cache directory should be excluded"


@pytest.mark.asyncio
async def test_uvproject_handler_with_project_install():
    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        with tempfile.TemporaryDirectory() as tmp_user_folder:
            user_folder = Path(tmp_user_folder)
            pyproject_file = user_folder / "pyproject.toml"
            pyproject_file.write_text("[project]\nname = 'test-project'\nversion='0.1.0'")
            uv_lock_file = user_folder / "uv.lock"
            uv_lock_file.write_text("lock content")

            # Create UVProject installing the whole project
            from flyte._image import UVProject

            uv_project = UVProject(
                pyproject=pyproject_file.absolute(),
                uvlock=uv_lock_file.absolute(),
                project_install_mode="install_project",
            )

            cache_dir = user_folder / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "temp.txt").write_text("temp")
            (user_folder / "memo.txt").write_text("memo")
            (user_folder / "main.py").write_text("print('hello')")

            initial_dockerfile = "FROM python:3.9\n"
            result = await UVProjectHandler.handle(
                layer=uv_project,
                context_path=context_path,
                dockerfile=initial_dockerfile,
                docker_ignore_patterns=["*.txt", ".cache", "pyproject.toml", "*.toml", "uv.lock", "*.lock"],
            )

            assert result.startswith(initial_dockerfile)
            assert "RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv" in result
            assert "uv sync" in result

            # Calculate expected destination path
            src_absolute = user_folder.absolute()
            rel_path = PurePath(*src_absolute.parts[1:])
            expected_dst_path = context_path / "_flyte_abs_context" / rel_path

            # Verify directory copy results and file exclusions
            assert expected_dst_path.exists(), f"Directory should be copied to {expected_dst_path}"
            assert expected_dst_path.is_dir(), "Should be a directory"
            assert (expected_dst_path / "main.py").exists(), "main.py should be included"
            assert (expected_dst_path / "pyproject.toml").exists(), "pyproject.toml should be included"
            assert (expected_dst_path / "uv.lock").exists(), "uv.lock should be included"
            assert not (expected_dst_path / "memo.txt").exists(), "memo.txt should be excluded"
            assert not (expected_dst_path / ".cache").exists(), ".cache directory should be excluded"


@pytest_asyncio.fixture
async def uv_project_with_editable(tmp_path: Path):
    """An empty uv project with a single editable dependency"""

    async def _uv(cmd: list[str], cwd: Path):
        return await asyncio.to_thread(
            subprocess.run, ["uv", *cmd], cwd=str(cwd), capture_output=True, text=True, check=True
        )

    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    # Create a main project
    await _uv(["init", "--lib"], project_root)
    # Create an editable dependency
    dep_folder = project_root / "libs" / "editable_dep"
    dep_folder.mkdir(parents=True)
    # Create an editable dependency project and add it to the main project
    await _uv(["init", "--lib"], dep_folder)
    await _uv(["add", "--editable", "./libs/editable_dep", "--no-sync"], project_root)
    # Generate a lock file for the main project
    await _uv(["lock"], project_root)
    yield project_root, dep_folder


@pytest.mark.asyncio
async def test_uvproject_handler_includes_editable_mounts_in_dependencies_only_mode(uv_project_with_editable):
    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        project_root, dep_folder = uv_project_with_editable
        pyproject_file = project_root / "pyproject.toml"
        uv_lock_file = project_root / "uv.lock"

        uv_project = UVProject(
            pyproject=pyproject_file.absolute(),
            uvlock=uv_lock_file.absolute(),
            project_install_mode="dependencies_only",
        )

        initial_dockerfile = "FROM python:3.9\n"
        result = await UVProjectHandler.handle(
            layer=uv_project,
            context_path=context_path,
            dockerfile=initial_dockerfile,
            docker_ignore_patterns=[],
        )
        expected_dep_in_context = "_flyte_abs_context" + str(dep_folder)
        expected_dep_in_container = dep_folder.relative_to(project_root)
        expected_mount = f"--mount=type=bind,src={expected_dep_in_context},target={expected_dep_in_container}"
        assert expected_mount in result


@pytest.mark.asyncio
async def test_uvproject_handler_without_uvlock():
    """Test that UVProjectHandler works correctly when uvlock is None."""
    with tempfile.TemporaryDirectory() as tmp_context, tempfile.TemporaryDirectory() as tmp_user:
        context_path = Path(tmp_context)
        user_folder = Path(tmp_user)

        # Create a pyproject.toml but no uv.lock file
        pyproject_file = user_folder / "pyproject.toml"
        pyproject_file.write_text("[project]\nname = 'test-project'\nversion='0.1.0'")

        # Create UVProject without uvlock
        uv_project = UVProject(
            pyproject=pyproject_file.absolute(),
            uvlock=None,
            project_install_mode="dependencies_only",
        )

        initial_dockerfile = "FROM python:3.9\n"
        result = await UVProjectHandler.handle(
            layer=uv_project,
            context_path=context_path,
            dockerfile=initial_dockerfile,
            docker_ignore_patterns=[],
        )

        # Verify the dockerfile is generated correctly
        assert result.startswith(initial_dockerfile)
        assert "RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv" in result
        assert "uv sync" in result
        # Verify that uvlock mount is NOT present
        assert "--mount=type=bind,target=uv.lock" not in result
        # Verify pyproject mount IS present
        assert "--mount=type=bind,target=pyproject.toml" in result


def test_get_secret_commands_deduplicates_secrets(monkeypatch):
    """Test that _get_secret_commands does not add duplicate secrets."""
    monkeypatch.setenv("GITHUB_TOKEN", "test-value")

    # Create layers with the same secret used multiple times
    same_secret = Secret(key="github_token")
    layers = (
        AptPackages(packages=("git", "vim"), secret_mounts=(same_secret,)),
        PipPackages(packages=("requests",), secret_mounts=(same_secret,)),
        Commands(commands=("echo hello",), secret_mounts=(same_secret,)),
    )

    commands = _get_secret_commands(layers)

    # Count how many times the secret appears in commands
    secret_count = sum(1 for cmd in commands if cmd == "--secret")
    assert secret_count == 1, f"Expected 1 secret, got {secret_count}. Commands: {commands}"


def test_get_secret_commands_allows_different_secrets(monkeypatch):
    """Test that _get_secret_commands allows different secrets."""
    monkeypatch.setenv("SECRET_A", "value-a")
    monkeypatch.setenv("SECRET_B", "value-b")

    secret_a = Secret(key="secret_a")
    secret_b = Secret(key="secret_b")
    layers = (
        AptPackages(packages=("git",), secret_mounts=(secret_a,)),
        PipPackages(packages=("requests",), secret_mounts=(secret_b,)),
    )

    commands = _get_secret_commands(layers)

    # Should have 2 different secrets
    secret_count = sum(1 for cmd in commands if cmd == "--secret")
    assert secret_count == 2, f"Expected 2 secrets, got {secret_count}. Commands: {commands}"


def test_get_secret_commands_deduplicates_string_secrets(monkeypatch):
    """Test that _get_secret_commands deduplicates string-based secrets."""
    monkeypatch.setenv("MY_TOKEN", "test-value")

    layers = (
        AptPackages(packages=("git",), secret_mounts=("my_token",)),
        PipPackages(packages=("requests",), secret_mounts=("my_token",)),
    )

    commands = _get_secret_commands(layers)

    secret_count = sum(1 for cmd in commands if cmd == "--secret")
    assert secret_count == 1, f"Expected 1 secret, got {secret_count}. Commands: {commands}"


def test_get_secret_commands_deduplicates_with_group(monkeypatch):
    """Test that _get_secret_commands deduplicates secrets with the same group and key."""
    monkeypatch.setenv("MYGROUP_MYKEY", "test-value")

    same_secret = Secret(group="mygroup", key="mykey")
    layers = (
        AptPackages(packages=("git",), secret_mounts=(same_secret,)),
        PipPackages(packages=("requests",), secret_mounts=(same_secret,)),
    )

    commands = _get_secret_commands(layers)

    secret_count = sum(1 for cmd in commands if cmd == "--secret")
    assert secret_count == 1, f"Expected 1 secret, got {secret_count}. Commands: {commands}"


def test_get_build_secrets_from_image_deduplicates_secrets():
    """Test that _get_build_secrets_from_image does not add duplicate secrets."""
    same_secret = Secret(key="github_token")

    image = (
        Image.from_debian_base(registry="localhost:30000", name="test", install_flyte=False)
        .with_apt_packages("git", "vim", secret_mounts=same_secret)
        .with_pip_packages("requests", secret_mounts=same_secret)
        .with_commands(["echo hello"], secret_mounts=same_secret)
    )

    secrets = _get_build_secrets_from_image(image)

    # Should only have 1 secret, not 3
    assert len(secrets) == 1, f"Expected 1 secret, got {len(secrets)}. Secrets: {secrets}"
    assert secrets[0].key == "github_token"


def test_get_build_secrets_from_image_allows_different_secrets():
    """Test that _get_build_secrets_from_image allows different secrets."""
    secret_a = Secret(key="secret_a")
    secret_b = Secret(key="secret_b")

    image = (
        Image.from_debian_base(registry="localhost:30000", name="test", install_flyte=False)
        .with_apt_packages("git", secret_mounts=secret_a)
        .with_pip_packages("requests", secret_mounts=secret_b)
    )

    secrets = _get_build_secrets_from_image(image)

    assert len(secrets) == 2, f"Expected 2 secrets, got {len(secrets)}. Secrets: {secrets}"
    keys = {s.key for s in secrets}
    assert keys == {"secret_a", "secret_b"}


def test_get_build_secrets_from_image_deduplicates_string_secrets():
    """Test that _get_build_secrets_from_image deduplicates string-based secrets."""
    image = (
        Image.from_debian_base(registry="localhost:30000", name="test", install_flyte=False)
        .with_apt_packages("git", secret_mounts="my_token")
        .with_pip_packages("requests", secret_mounts="my_token")
    )

    secrets = _get_build_secrets_from_image(image)

    assert len(secrets) == 1, f"Expected 1 secret, got {len(secrets)}. Secrets: {secrets}"
    assert secrets[0].key == "my_token"


def test_get_build_secrets_from_image_deduplicates_with_group():
    """Test that _get_build_secrets_from_image deduplicates secrets with the same group and key."""
    same_secret = Secret(group="mygroup", key="mykey")

    image = (
        Image.from_debian_base(registry="localhost:30000", name="test", install_flyte=False)
        .with_apt_packages("git", secret_mounts=same_secret)
        .with_pip_packages("requests", secret_mounts=same_secret)
    )

    secrets = _get_build_secrets_from_image(image)

    assert len(secrets) == 1, f"Expected 1 secret, got {len(secrets)}. Secrets: {secrets}"
    assert secrets[0].key == "mykey"
    assert secrets[0].group == "mygroup"


def test_uv_base_template_default_venv():
    """When base image has no UV_PYTHON, the template should default to /opt/venv and create a venv."""
    dockerfile = DOCKER_FILE_UV_BASE_TEMPLATE.substitute(
        BASE_IMAGE="python:3.12-slim",
        PYTHON_VERSION="3.12",
    )

    # Should declare default paths via ARG
    assert "ARG VIRTUALENV=/opt/venv" in dockerfile
    assert "ARG UV_PYTHON=$VIRTUALENV/bin/python" in dockerfile

    # Should set ENV from ARGs
    assert "VIRTUALENV=$VIRTUALENV" in dockerfile
    assert "UV_PYTHON=$UV_PYTHON" in dockerfile

    # Should conditionally create venv only if UV_PYTHON binary doesn't exist
    assert 'if [ ! -f "$UV_PYTHON" ]' in dockerfile
    assert "uv venv $VIRTUALENV --python=3.12" in dockerfile

    # Should add VIRTUALENV/bin to PATH
    assert 'PATH="$VIRTUALENV/bin:$PATH"' in dockerfile


def test_uv_base_template_preserves_existing_uv_python():
    """When base image has UV_PYTHON set, the template should preserve it and skip venv creation."""
    dockerfile = DOCKER_FILE_UV_BASE_TEMPLATE.substitute(
        BASE_IMAGE="my-custom-image:latest",
        PYTHON_VERSION="3.12",
    )

    # UV_PYTHON ARG defaults to $VIRTUALENV/bin/python but can be overridden by base image
    assert "ARG UV_PYTHON=$VIRTUALENV/bin/python" in dockerfile

    # The conditional block skips venv creation when UV_PYTHON binary already exists
    assert 'if [ ! -f "$UV_PYTHON" ]' in dockerfile

    # PATH includes VIRTUALENV/bin
    assert 'PATH="$VIRTUALENV/bin:$PATH"' in dockerfile


@pytest.mark.asyncio
async def test_ensure_buildx_builder_creates_with_host_network():
    """When creating a new buildx builder, it should use --driver-opt network=host."""
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        result = subprocess.CompletedProcess(cmd, 0)
        # For 'docker buildx ls', return output without the builder name
        if cmd == ["docker", "buildx", "ls"]:
            result.stdout = "default"
            result.stderr = ""
        return result

    with patch(
        "flyte._internal.imagebuild.docker_builder.run_sync_with_loop", side_effect=lambda fn, *a, **kw: fn(*a, **kw)
    ):
        with patch("subprocess.run", side_effect=mock_run):
            await DockerImageBuilder._ensure_buildx_builder()

    # Find the create command
    create_cmds = [c for c in calls if "create" in c]
    assert len(create_cmds) == 1
    create_cmd = create_cmds[0]
    assert "--driver-opt" in create_cmd
    assert "network=host" in create_cmd


@pytest.mark.asyncio
async def test_ensure_buildx_builder_skips_when_network_host_present():
    """When the builder already exists with network=host, it should not recreate."""
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        result = subprocess.CompletedProcess(cmd, 0)
        if cmd == ["docker", "buildx", "ls"]:
            result.stdout = f"default\n{DockerImageBuilder._builder_name}  docker-container"
            result.stderr = ""
        elif "inspect" in cmd:
            result.stdout = (
                f"Name:          {DockerImageBuilder._builder_name}\n"
                "Driver:        docker-container\n"
                "Nodes:\n"
                'Driver Options: network="host"\n'
            )
            result.stderr = ""
        return result

    with patch(
        "flyte._internal.imagebuild.docker_builder.run_sync_with_loop", side_effect=lambda fn, *a, **kw: fn(*a, **kw)
    ):
        with patch("subprocess.run", side_effect=mock_run):
            await DockerImageBuilder._ensure_buildx_builder()

    # Should NOT have called create or rm
    assert not any("create" in c for c in calls)
    assert not any("rm" in c for c in calls)


@pytest.mark.asyncio
async def test_ensure_buildx_builder_recreates_when_network_host_missing():
    """When the builder exists but is missing network=host, it should be removed and recreated."""
    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        result = subprocess.CompletedProcess(cmd, 0)
        if cmd == ["docker", "buildx", "ls"]:
            result.stdout = f"default\n{DockerImageBuilder._builder_name}  docker-container"
            result.stderr = ""
        elif "inspect" in cmd:
            result.stdout = (
                f"Name:          {DockerImageBuilder._builder_name}\n"
                "Driver:        docker-container\n"
                "Nodes:\n"
                "Driver Options: <none>\n"
            )
            result.stderr = ""
        return result

    with patch(
        "flyte._internal.imagebuild.docker_builder.run_sync_with_loop", side_effect=lambda fn, *a, **kw: fn(*a, **kw)
    ):
        with patch("subprocess.run", side_effect=mock_run):
            await DockerImageBuilder._ensure_buildx_builder()

    # Should have called rm then create
    rm_cmds = [c for c in calls if "rm" in c]
    create_cmds = [c for c in calls if "create" in c]
    assert len(rm_cmds) == 1
    assert DockerImageBuilder._builder_name in rm_cmds[0]
    assert len(create_cmds) == 1
    assert "--driver-opt" in create_cmds[0]
    assert "network=host" in create_cmds[0]


@pytest.mark.asyncio
async def test_build_image_uses_custom_builder_from_env(monkeypatch):
    """When FLYTE_DOCKER_BUILDER_NAME is set, _build_image should use it and skip _ensure_buildx_builder."""
    from flyte._internal.imagebuild import docker_builder as db

    monkeypatch.setenv("FLYTE_DOCKER_BUILDER_NAME", "my-custom-builder")

    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    ensure_called = False

    async def fake_ensure():
        nonlocal ensure_called
        ensure_called = True

    img = Image.from_debian_base(registry="localhost:30000", name="custom_builder_test", install_flyte=False)

    with patch.object(db.DockerImageBuilder, "_ensure_buildx_builder", side_effect=fake_ensure):
        with patch(
            "flyte._internal.imagebuild.docker_builder.run_sync_with_loop",
            side_effect=lambda fn, *a, **kw: fn(*a, **kw),
        ):
            with patch("subprocess.run", side_effect=mock_run):
                await db.DockerImageBuilder()._build_image(img, push=False, dry_run=False)

    assert ensure_called is False
    build_cmds = [c for c in calls if isinstance(c, list) and "build" in c and "buildx" in c]
    assert build_cmds, "expected a buildx build command"
    cmd = build_cmds[0]
    builder_idx = cmd.index("--builder")
    assert cmd[builder_idx + 1] == "my-custom-builder"


@pytest.mark.asyncio
async def test_build_image_uses_default_builder_when_env_unset(monkeypatch):
    """When FLYTE_DOCKER_BUILDER_NAME is unset, _build_image should call _ensure_buildx_builder and use the default name."""
    from flyte._internal.imagebuild import docker_builder as db

    monkeypatch.delenv("FLYTE_DOCKER_BUILDER_NAME", raising=False)

    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    ensure_called = False

    async def fake_ensure():
        nonlocal ensure_called
        ensure_called = True

    img = Image.from_debian_base(registry="localhost:30000", name="default_builder_test", install_flyte=False)

    with patch.object(db.DockerImageBuilder, "_ensure_buildx_builder", side_effect=fake_ensure):
        with patch(
            "flyte._internal.imagebuild.docker_builder.run_sync_with_loop",
            side_effect=lambda fn, *a, **kw: fn(*a, **kw),
        ):
            with patch("subprocess.run", side_effect=mock_run):
                await db.DockerImageBuilder()._build_image(img, push=False, dry_run=False)

    assert ensure_called is True
    build_cmds = [c for c in calls if isinstance(c, list) and "build" in c and "buildx" in c]
    assert build_cmds
    cmd = build_cmds[0]
    builder_idx = cmd.index("--builder")
    assert cmd[builder_idx + 1] == db.DockerImageBuilder._builder_name


@pytest.mark.asyncio
async def test_build_from_dockerfile_uses_custom_builder_from_env(monkeypatch):
    """_build_from_dockerfile should respect FLYTE_DOCKER_BUILDER_NAME and skip ensure when set."""
    from flyte._internal.imagebuild import docker_builder as db

    monkeypatch.setenv("FLYTE_DOCKER_BUILDER_NAME", "my-custom-builder")

    calls = []

    def mock_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    ensure_called = False

    async def fake_ensure():
        nonlocal ensure_called
        ensure_called = True

    with tempfile.TemporaryDirectory() as tmp_dir:
        dockerfile = Path(tmp_dir) / "Dockerfile"
        dockerfile.write_text("FROM python:3.12\n")

        img = Image.from_dockerfile(
            file=dockerfile, registry="localhost:30000", name="custom_dockerfile_test"
        )

        with patch.object(db.DockerImageBuilder, "_ensure_buildx_builder", side_effect=fake_ensure):
            with patch(
                "flyte._internal.imagebuild.docker_builder.run_sync_with_loop",
                side_effect=lambda fn, *a, **kw: fn(*a, **kw),
            ):
                with patch("subprocess.run", side_effect=mock_run):
                    await db.DockerImageBuilder()._build_from_dockerfile(img, push=False)

    assert ensure_called is False
    build_cmds = [c for c in calls if isinstance(c, list) and "build" in c and "buildx" in c]
    assert build_cmds
    cmd = build_cmds[0]
    builder_idx = cmd.index("--builder")
    assert cmd[builder_idx + 1] == "my-custom-builder"


def test_dockerfile_footer_switches_to_flyte_user():
    """The footer should switch the runtime user to flyte and set the workdir to /home/flyte."""
    from flyte._internal.imagebuild.docker_builder import DOCKER_FILE_BASE_FOOTER

    rendered = DOCKER_FILE_BASE_FOOTER.substitute(F_IMG_ID="some-image-id")

    assert "USER flyte" in rendered
    assert "WORKDIR /home/flyte" in rendered
    assert "ENV _F_IMG_ID=some-image-id" in rendered


@pytest.mark.asyncio
async def test_copy_config_handler_uses_chown_flyte():
    """CopyConfigHandler should emit COPY --chown=flyte:flyte so the runtime user owns the
    files added via with_source_file / with_source_folder."""
    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        with tempfile.TemporaryDirectory() as tmp_src_dir:
            src_dir = Path(tmp_src_dir)
            test_file = src_dir / "main.py"
            test_file.write_text("print('hello')")

            copy_config = CopyConfig(
                src=test_file,
                dst="/home/flyte/main.py",
                path_type=0,
            )

            result = await CopyConfigHandler.handle(
                layer=copy_config,
                context_path=context_path,
                dockerfile="FROM python:3.12\n",
                docker_ignore_patterns=[],
            )

            assert "COPY --chown=flyte:flyte" in result
            assert "/home/flyte/main.py" in result


@pytest.mark.asyncio
async def test_code_bundle_handler_uses_chown_flyte():
    """_CodeBundleHandler should emit COPY --chown=flyte:flyte for code bundles baked into the image."""
    from flyte._image import CodeBundleLayer
    from flyte._internal.imagebuild.docker_builder import _CodeBundleHandler

    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        with tempfile.TemporaryDirectory() as tmp_src_dir:
            src_dir = Path(tmp_src_dir)
            (src_dir / "task.py").write_text("print('task')")

            layer = CodeBundleLayer(copy_style="all", dst="/home/flyte/code", root_dir=src_dir)

            result = await _CodeBundleHandler.handle(
                layer=layer,
                context_path=context_path,
                dockerfile="FROM python:3.12\n",
            )

            assert "COPY --chown=flyte:flyte" in result
            assert "/home/flyte/code" in result
