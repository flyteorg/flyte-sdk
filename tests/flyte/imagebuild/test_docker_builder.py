import tempfile
from pathlib import Path

import pytest

from flyte import Secret
from flyte._image import Image, PipPackages, PoetryProject, Requirements
from flyte._internal.imagebuild.docker_builder import (
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
    )

    builder = DockerImageBuilder()
    await builder.build_image(img)


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
                extra_index_urls=["--no-install-project"],
                secret_mounts=None,
            )

            initial_dockerfile = "FROM python:3.9\n"
            result = await PoetryProjectHandler.handel(poetry_project, context_path, initial_dockerfile)

            assert "RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv" in result
            assert "--mount=type=bind,target=uv.lock,src=" in result
            assert "--mount=type=bind,target=pyproject.toml,src=" in result
            assert "uv pip install poetry" in result
            assert "ENV POETRY_CACHE_DIR=/tmp/poetry_cache" in result
            assert "POETRY_VIRTUALENVS_IN_PROJECT=true" in result
            assert "WORKDIR /root" in result
            assert "poetry install --no-root" in result

            # Should not contain COPY command for entire project
            assert "COPY" not in result


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

            # Create PoetryProject without --no-install-project flag
            poetry_project = PoetryProject(pyproject=pyproject_file.absolute(), poetry_lock=None)

            initial_dockerfile = "FROM python:3.9\n"
            result = await PoetryProjectHandler.handel(poetry_project, context_path, initial_dockerfile)

            assert result.startswith(initial_dockerfile)

            assert "COPY" in result
            assert "pyproject.toml" in result
            assert "RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv" in result
            assert "uv pip install poetry" in result
            assert "ENV POETRY_CACHE_DIR=/tmp/poetry_cache" in result
            assert "POETRY_VIRTUALENVS_IN_PROJECT=true" in result
            assert "WORKDIR /root" in result
            assert "poetry install --no-root" in result
