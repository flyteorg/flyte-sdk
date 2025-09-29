import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from flyte import Image
from flyte._internal.imagebuild.utils import copy_files_to_context, get_and_list_dockerignore


def test_get_and_list_dockerignore_with_dockerignore_file():
    with tempfile.TemporaryDirectory() as tmp_context:
        src_dir = Path(tmp_context)
        dockerignore_file = src_dir / ".dockerignore"
        dockerignore_file.write_text("*.py\nsrc/\n.cache\n# This is a comment\n\n*.txt\n \n  \n\t\n")
        image = Image.from_debian_base()

        # Mock _get_init_config to return src_dir as root_dir
        mock_config = MagicMock()
        mock_config.root_dir = src_dir

        with patch("flyte._initialize._get_init_config", return_value=mock_config):
            patterns = get_and_list_dockerignore(image)
            expected_patterns = ["*.py", "src/", ".cache", "*.txt"]
            assert patterns == expected_patterns


def test_get_and_list_dockerignore_with_dockerignore_layer():
    with tempfile.TemporaryDirectory() as tmp_context:
        src_dir = Path(tmp_context)
        root_dockerignore = src_dir / ".dockerignore"
        root_dockerignore.write_text("*.py\nsrc/\n")
        custom_dockerignore = src_dir / "custom.dockerignore"
        custom_dockerignore.write_text("*.txt\n.cache\n")
        image = Image.from_debian_base().with_dockerignore(custom_dockerignore)
        patterns = get_and_list_dockerignore(image)
        expected_patterns = ["*.txt", ".cache"]

        assert patterns == expected_patterns


def test_get_and_list_dockerignore_not_found():
    with tempfile.TemporaryDirectory() as tmp_context:
        src_dir = Path(tmp_context)
        image = Image.from_debian_base()

        # Mock _get_init_config to return src_dir as root_dir
        mock_config = MagicMock()
        mock_config.root_dir = src_dir

        with patch("flyte._initialize._get_init_config", return_value=mock_config):
            patterns = get_and_list_dockerignore(image)
            assert patterns == []


def test_get_and_list_dockerignore_layer_priority():
    with tempfile.TemporaryDirectory() as tmp_context:
        src_dir = Path(tmp_context)
        local_dockerignore = src_dir / ".dockerignore"
        local_dockerignore.write_text("*.py\nsrc/\n")
        layer_dockerignore = src_dir / "custom.dockerignore"
        layer_dockerignore.write_text("*.txt\n.cache\n")
        image = Image.from_debian_base().with_dockerignore(layer_dockerignore)

        # Mock _get_init_config to return src_dir as root_dir
        mock_config = MagicMock()
        mock_config.root_dir = src_dir

        with patch("flyte._initialize._get_init_config", return_value=mock_config):
            patterns = get_and_list_dockerignore(image)
            expected_patterns = ["*.txt", ".cache"]
            assert patterns == expected_patterns


def test_copy_files_to_context_protected_patterns():
    """Test that protected patterns are not excluded even when they match ignore patterns"""
    with tempfile.TemporaryDirectory() as tmp_context:
        context_path = Path(tmp_context)
        with tempfile.TemporaryDirectory() as tmp_user_folder:
            src_dir = Path(tmp_user_folder)

            pyproject_file = src_dir / "pyproject.toml"
            pyproject_file.write_text("[project]\nname = 'test'")
            other_toml = src_dir / "other.toml"
            other_toml.write_text("[config]\nvalue = 1")
            uv_lock = src_dir / "uv.lock"
            uv_lock.write_text("lock file content")

            copy_files_to_context(
                src=src_dir,
                context_path=context_path,
                ignore_patterns=["*.toml", "pyproject.toml", "*.lock", "uv.lock"],
                protected_patterns=["pyproject.toml", "uv.lock"],
            )

            # Calculate expected destination path
            src_absolute = src_dir.absolute()
            dst_path_str = str(src_absolute).replace("/", "./_flyte_abs_context/", 1)
            expected_dst_path = context_path / dst_path_str

            pyproject_exists = (expected_dst_path / "pyproject.toml").exists()
            assert pyproject_exists
            other_toml_exists = (expected_dst_path / "other.toml").exists()
            assert not other_toml_exists
            uv_lock_exists = (expected_dst_path / "uv.lock").exists()
            assert uv_lock_exists
