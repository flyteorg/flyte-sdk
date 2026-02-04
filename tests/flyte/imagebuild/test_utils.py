import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from flyte import Image
from flyte._internal.imagebuild.utils import (
    get_and_list_dockerignore,
    get_uv_editable_install_mounts,
    get_uv_project_editable_dependencies,
)


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


def test_get_uv_lock_editable_dependencies_resolves_paths():
    with tempfile.TemporaryDirectory() as tmp_context:
        project_root = Path(tmp_context)
        editable_abs = project_root / "editable-abs"
        editable_rel = "editable-rel"
        editable_parent = "../editable-parent"
        expected_abs = editable_abs
        expected_rel = project_root / editable_rel
        expected_parent = project_root / editable_parent

        export_output = "\n".join(
            [
                f"-e {editable_abs}",
                f"-e {editable_rel}",
                f"-e {editable_parent}",
                "somepkg==1.2.3",
            ]
        )
        mock_result = MagicMock(stdout=export_output)
        with patch("flyte._internal.imagebuild.utils.subprocess.run", return_value=mock_result):
            paths = get_uv_project_editable_dependencies(project_root)

        assert paths == [expected_abs, expected_rel, expected_parent]


def test_get_uv_editable_install_mounts():
    with tempfile.TemporaryDirectory() as tmp_context:
        project_root = Path(tmp_context) / "project"
        context_path = Path(tmp_context) / "context"
        project_root.mkdir(parents=True)
        context_path.mkdir(parents=True)

        editable_abs = str(project_root / "editable-abs")
        editable_rel = "./editable-rel"

        # Create the editable dependencies
        for path in (Path(editable_abs), Path(project_root / editable_rel)):
            os.makedirs(path, exist_ok=True)

        with patch(
            "flyte._internal.imagebuild.utils._extract_editables_from_uv_export",
            return_value=[editable_abs, editable_rel],
        ):
            mounts = get_uv_editable_install_mounts(project_root, context_path, ignore_patterns=[])

        # NOTE: If any library at path <PATH> should be expected under _flyte_abs_context/<PATH>.
        # wihtin the build context.
        # However, within the container, it should be mounted to its path relative to the project root.
        # This is done by using the relative_to method on the Path objects.
        expected_mounts = [
            f"--mount=type=bind,src=_flyte_abs_context{editable_abs},target={Path(editable_abs).relative_to(project_root)}",
            f"--mount=type=bind,src=_flyte_abs_context{project_root / editable_rel},target={Path(editable_rel).name}",
        ]
        assert mounts == " ".join(expected_mounts)
