import asyncio
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
async def test_copy_config_handler_handle_adds_copy_command():
    """Test that CopyConfigHandler.handle method correctly adds COPY command to dockerfile"""
    from flyte._image import CopyConfig

    # Create temporary directory as context path
    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        # Create temporary source directory
        with tempfile.TemporaryDirectory() as src_tmpdir:
            src_path = Path(src_tmpdir)
            # Create test file
            (src_path / "test_file.txt").write_text("test content")

            # Create CopyConfig
            layer = CopyConfig(path_type=1, src=src_path, dst="/app")
            dockerfile = "FROM python:3.9"

            # Call handle method
            result = await CopyConfigHandler.handle(layer, context_path, dockerfile)

            # Verify results
            assert "COPY" in result
            assert src_path.name in result
            assert "/app" in result


@pytest.mark.asyncio
async def test_copy_config_handler_handle_no_copy_when_empty():
    """Test that no COPY command is added when no files are copied"""
    from flyte._image import CopyConfig

    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)

        with tempfile.TemporaryDirectory() as src_tmpdir:
            src_path = Path(src_tmpdir)
            # Create empty directory, no files

            layer = CopyConfig(path_type=1, src=src_path, dst="/app")
            dockerfile = "FROM python:3.9"

            result = await CopyConfigHandler.handle(layer, context_path, dockerfile)

            # Verify no COPY command is added
            assert "COPY" not in result
            # Verify original dockerfile content remains unchanged
            assert result == "FROM python:3.9"
            

@pytest.mark.asyncio
async def test_copy_files_recursively_single_file():
    """Test copy_files_recursively method with a single file source."""
    with tempfile.TemporaryDirectory() as tmp_dst:
        dst_path = Path(tmp_dst)
        
        # Create a temporary file as source
        src_file_path = Path(tempfile.mktemp(suffix='.txt'))
        src_file_path.write_text("test content")
        
        # Test copying single file with path_type=0
        empty_ignore = IgnoreGroup(src_file_path.parent)
        
        copied_files = CopyConfigHandler.copy_files_recursively(
            src_file_path, dst_path, 0, deref_symlinks=False, ignore_group=empty_ignore
        )
        
        # Verify file was copied
        assert len(copied_files) == 1
        assert copied_files[0] == dst_path.name  # Should be the destination directory name
        
        # Verify file actually exists in destination
        expected_dst_file = dst_path / src_file_path.name
        assert expected_dst_file.exists()


@pytest.mark.asyncio
async def test_copy_files_recursively_single_file_with_ignore():
    """Test copy_files_recursively method with a single file source that should be ignored."""
    with tempfile.TemporaryDirectory() as tmp_dst:
        dst_path = Path(tmp_dst)
        
        # Create a temporary file as source
        src_file_path = Path(tempfile.mktemp(suffix='.pyc'))
        src_file_path.write_text("test content")
        
        # Test copying single file that should be ignored
        ignore_group = IgnoreGroup(src_file_path.parent, StandardIgnore)
        
        copied_files = CopyConfigHandler.copy_files_recursively(
            src_file_path, dst_path, 0, deref_symlinks=False, ignore_group=ignore_group
        )
        
        # Verify file was not copied due to ignore pattern
        assert len(copied_files) == 0
        
        # Verify file does not exist in destination
        expected_dst_file = dst_path / src_file_path.name
        assert not expected_dst_file.exists()


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
        copied_files = CopyConfigHandler.copy_files_recursively(src_path, dst_path, 1, deref_symlinks=False, ignore_group=empty_ignore)

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
        copied_files = CopyConfigHandler.copy_files_recursively(src_path, dst_path, 1, deref_symlinks=False, ignore_group=ignore_group)

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

        # GitIgnore requires a real git repository to work properly
        # So we need to initialize git and commit files
        import subprocess

        async def run_git_command(cmd, cwd):
            """Execute git command asynchronously"""
            process = await asyncio.create_subprocess_exec(
                *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

        try:
            # Initialize git repository
            await run_git_command(["git", "init"], src_path)

            # Add all files
            await run_git_command(["git", "add", "."], src_path)

            # Configure git user
            await run_git_command(["git", "config", "user.name", "test"], src_path)
            await run_git_command(["git", "config", "user.email", "test@example.com"], src_path)

            # Commit files
            await run_git_command(["git", "commit", "-m", "initial"], src_path)

            # Create ignore group with GitIgnore
            ignore_group = GitIgnore(src_path)

            # Test with gitignore
            copied_files = CopyConfigHandler.copy_files_recursively(
                src_path, dst_path, 1, deref_symlinks=False, ignore_group=ignore_group
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

        except (subprocess.CalledProcessError, FileNotFoundError):
            # If git is not available or fails, skip this test
            pytest.skip("Git is not available or failed, skipping gitignore test")


@pytest.mark.asyncio
async def test_copy_files_recursively_file_not_found():
    """Test copy_files_recursively method when source file/directory does not exist."""
    with tempfile.TemporaryDirectory() as tmp_dst:
        dst_path = Path(tmp_dst)
        
        # Create a non-existent source path
        non_existent_path = Path("/non/existent/path")
        
        # Test with non-existent file path_type=0
        from flyte._code_bundle._ignore import IgnoreGroup
        empty_ignore = IgnoreGroup(Path("/"))
        
        copied_files = CopyConfigHandler.copy_files_recursively(
            non_existent_path, dst_path, 0, deref_symlinks=False, ignore_group=empty_ignore
        )
        
        # Verify empty list is returned when source does not exist
        assert len(copied_files) == 0
        assert copied_files == []
        
        # Test with non-existent directory path_type=1
        copied_files = CopyConfigHandler.copy_files_recursively(
            non_existent_path, dst_path, 1, deref_symlinks=False, ignore_group=empty_ignore
        )
        
        # Verify empty list is returned when source does not exist
        assert len(copied_files) == 0
        assert copied_files == []
