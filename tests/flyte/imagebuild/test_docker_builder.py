import tempfile
from pathlib import Path

import pytest

from flyte import Secret
from flyte._code_bundle._ignore import GitIgnore, IgnoreGroup, StandardIgnore
from flyte._image import Image, PipPackages, Requirements
from flyte._internal.imagebuild.docker_builder import CopyConfigHandler, DockerImageBuilder, PipAndRequirementsHandler


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
    secret1, _ = tempfile.mkstemp("secret1")
    monkeypatch.setenv("GROUP_KEY", "test-value")

    img = (
        Image.from_debian_base(registry="localhost:30000", name="img_with_secrets")
        .with_apt_packages("vim", secret_mounts=[str(secret1)])
        .with_pip_packages("requests", secret_mounts=[Secret(group="group", key="key")])
    )

    builder = DockerImageBuilder()
    await builder.build_image(img)


@pytest.mark.asyncio
async def test_pip_package_handling(monkeypatch):
    secret1, _ = tempfile.mkstemp("secret1")
    monkeypatch.setenv("GROUP_KEY", "test-value")

    # Create a temporary directory to simulate the context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_path = Path(tmpdir)

        # raw pip packages
        pip_packages = PipPackages(packages=("pkg_a", "pkg_b"), secret_mounts=[str(secret1)])
        docker_update = await PipAndRequirementsHandler.handle(
            layer=pip_packages, context_path=context_path, dockerfile=""
        )
        assert "--mount=type=secret" in docker_update
        assert "uv pip install --python $UV_PYTHON pkg_a pkg_b" in docker_update


@pytest.mark.asyncio
async def test_requirements_handler(monkeypatch):
    secret1, _ = tempfile.mkstemp("secret1")
    monkeypatch.setenv("GROUP_KEY", "test-value")

    # Create a temporary directory to simulate the context
    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        with tempfile.TemporaryDirectory() as tmp_user_folder:
            user_folder = Path(tmp_user_folder)
            # create a dummy requirements.txt file
            requirements_file = user_folder / "requirements.txt"
            requirements_file.write_text("pkg_a\npkg_b\n")

            requirements = Requirements(file=requirements_file.absolute(), secret_mounts=[str(secret1)])
            docker_update = await PipAndRequirementsHandler.handle(
                layer=requirements, context_path=context_path, dockerfile=""
            )
            assert "--mount=type=secret" in docker_update
            assert "_flyte_abs_context" + str(requirements_file.absolute()) in docker_update


@pytest.mark.asyncio
async def test_copy_files_recursively_without_ignore():
    """Test copy_files_recursively method without ignore group."""
    with tempfile.TemporaryDirectory() as tmp_src, tempfile.TemporaryDirectory() as tmp_dst:
        src_path = Path(tmp_src)
        dst_path = Path(tmp_dst)

        # Create test directory structure
        (src_path / "subdir").mkdir()
        (src_path / "subdir" / "nested").mkdir()

        # Create test files
        (src_path / "file1.txt").write_text("content1")
        (src_path / "subdir" / "file2.py").write_text("content2")
        (src_path / "subdir" / "nested" / "file3.json").write_text("content3")

        # Test without ignore group - create an empty ignore group that doesn't ignore anything
        from flyte._code_bundle._ignore import IgnoreGroup

        empty_ignore = IgnoreGroup(src_path)
        copied_files = CopyConfigHandler.copy_files_recursively(src_path, dst_path, empty_ignore, deref_symlinks=False)

        # Verify all files were copied
        assert len(copied_files) == 3
        assert "file1.txt" in copied_files
        assert "subdir/file2.py" in copied_files
        assert "subdir/nested/file3.json" in copied_files

        # Verify files actually exist in destination
        assert (dst_path / "file1.txt").exists()
        assert (dst_path / "subdir" / "file2.py").exists()
        assert (dst_path / "subdir" / "nested" / "file3.json").exists()


@pytest.mark.asyncio
async def test_copy_files_recursively_with_ignore():
    """Test copy_files_recursively method with ignore group."""
    with tempfile.TemporaryDirectory() as tmp_src, tempfile.TemporaryDirectory() as tmp_dst:
        src_path = Path(tmp_src)
        dst_path = Path(tmp_dst)

        # Create test directory structure
        (src_path / "subdir").mkdir()
        (src_path / ".cache").mkdir()

        # Create test files
        (src_path / "file1.txt").write_text("content1")
        (src_path / "file2.pyc").write_text("content2")  # This should be ignored
        (src_path / "subdir" / "file3.py").write_text("content3")
        (src_path / ".cache" / "temp.dat").write_text("content4")

        ignores = (StandardIgnore, GitIgnore)
        ignore_group = IgnoreGroup(src_path, *ignores)

        # Test with ignore group
        copied_files = CopyConfigHandler.copy_files_recursively(src_path, dst_path, ignore_group, deref_symlinks=False)

        # Verify only non-ignored files were copied
        assert len(copied_files) == 2, f"Expected 2 files, but got {len(copied_files)}: {copied_files}"
        assert "file1.txt" in copied_files
        assert "subdir/file3.py" in copied_files

        # Verify ignored files were not copied
        assert "file2.pyc" not in copied_files
        assert ".cache/temp.dat" not in copied_files

        # Verify files actually exist/not exist in destination
        assert (dst_path / "file1.txt").exists()
        assert (dst_path / "subdir" / "file3.py").exists()
        assert not (dst_path / "file2.pyc").exists()
        assert not (dst_path / ".cache" / "temp.dat").exists()


@pytest.mark.asyncio
async def test_copy_files_recursively_with_gitignore():
    """Test copy_files_recursively method with gitignore patterns."""
    with tempfile.TemporaryDirectory() as tmp_src, tempfile.TemporaryDirectory() as tmp_dst:
        src_path = Path(tmp_src)
        dst_path = Path(tmp_dst)

        # Create test directory structure
        (src_path / "src").mkdir()
        (src_path / "tests").mkdir()

        # Create test files
        (src_path / "src" / "main.py").write_text("content1")
        (src_path / "tests" / "test_main.py").write_text("content2")
        (src_path / "README.md").write_text("content3")

        # Create .gitignore file
        gitignore_content = """
tests/
*.pyc
__pycache__/
        """.strip()
        (src_path / ".gitignore").write_text(gitignore_content)

        # Create git repository (simulate git init)
        import subprocess

        try:
            # Initialize git repository
            subprocess.run(["git", "init"], cwd=src_path, capture_output=True, check=True)

            # Add all files
            subprocess.run(["git", "add", "."], cwd=src_path, capture_output=True, check=True)

            # Configure git user
            subprocess.run(["git", "config", "user.name", "test"], cwd=src_path, capture_output=True, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"], cwd=src_path, capture_output=True, check=True
            )

            # Commit files
            subprocess.run(["git", "commit", "-m", "initial"], cwd=src_path, capture_output=True, check=True)

            # Create ignore group with GitIgnore
            ignore_group = GitIgnore(src_path)

            # Test with gitignore
            copied_files = CopyConfigHandler.copy_files_recursively(
                src_path, dst_path, ignore_group, deref_symlinks=False
            )

            # Verify only non-ignored files were copied
            # Note: .gitignore file itself is also copied, so we expect 3 files total
            assert len(copied_files) == 3, f"Expected 3 files, but got {len(copied_files)}: {copied_files}"
            assert "src/main.py" in copied_files
            assert "README.md" in copied_files
            assert ".gitignore" in copied_files

            # Verify ignored files were not copied
            assert "tests/test_main.py" not in copied_files

            # Verify files actually exist/not exist in destination
            assert (dst_path / "src" / "main.py").exists()
            assert (dst_path / "README.md").exists()
            assert not (dst_path / "tests" / "test_main.py").exists()

        except subprocess.CalledProcessError:
            # If git is not available, skip this test
            pytest.skip("Git is not available, skipping gitignore test")
