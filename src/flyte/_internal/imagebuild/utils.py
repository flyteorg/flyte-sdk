import re
import shutil
import subprocess
from pathlib import Path, PurePath
from typing import List, Optional

from flyte._code_bundle._ignore import STANDARD_IGNORE_PATTERNS
from flyte._image import DockerIgnore, Image
from flyte._logging import logger


def copy_files_to_context(src: Path, context_path: Path, ignore_patterns: list[str] = []) -> Path:
    """
    This helper function ensures that absolute paths that users specify are converted correctly to a path in the
    context directory. Doing this prevents collisions while ensuring files are available in the context.

    For example, if a user has
        img.with_requirements(Path("/Users/username/requirements.txt"))
           .with_requirements(Path("requirements.txt"))
           .with_requirements(Path("../requirements.txt"))

    copying with this function ensures that the Docker context folder has all three files.

    :param src: The source path to copy
    :param context_path: The context path where the files should be copied to
    """
    if src.is_absolute() or ".." in str(src):
        rel_path = PurePath(*src.parts[1:])
        dst_path = context_path / "_flyte_abs_context" / rel_path
    else:
        dst_path = context_path / src

    if src.is_dir():
        from .docker import PatternMatcher

        # Add ** prefix to match patterns anywhere in tree
        default_ignore_patterns = ["**/.idea", "**/.venv", "**/__pycache__", "**/*.pyc"]
        all_patterns = ignore_patterns + default_ignore_patterns
        pm = PatternMatcher(all_patterns)

        # Use walk() to get list of files to include
        for rel_file in pm.walk(str(src)):
            src_file = src / rel_file
            dst_file = dst_path / rel_file

            # Create parent directory if needed
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file (not directory)
            if src_file.is_file():
                shutil.copy2(src_file, dst_file)

    else:
        # Single file
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst_path)

    return dst_path


def get_and_list_dockerignore(image: Image) -> List[str]:
    """
    Get and parse dockerignore patterns from .dockerignore file.

    This function first looks for a DockerIgnore layer in the image's layers. If found, it uses
    the path specified in that layer. If no DockerIgnore layer is found, it falls back to looking
    for a .dockerignore file in the root_path directory.

    :param image: The Image object
    """
    from flyte._initialize import _get_init_config

    # Look for DockerIgnore layer in the image layers
    dockerignore_path: Optional[Path] = None
    patterns: List[str] = []

    for layer in image._layers:
        if isinstance(layer, DockerIgnore) and layer.path.strip():
            dockerignore_path = Path(layer.path)
    # If DockerIgnore layer not specified, set dockerignore_path under root_path
    init_config = _get_init_config()
    root_path = init_config.root_dir if init_config else None
    if not dockerignore_path and root_path:
        dockerignore_path = Path(root_path) / ".dockerignore"
    # Return empty list if no .dockerignore file found
    if not dockerignore_path or not dockerignore_path.exists() or not dockerignore_path.is_file():
        logger.info(f".dockerignore file not found at path: {dockerignore_path}")
        return patterns

    try:
        with open(dockerignore_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                # Skip empty lines, whitespace-only lines, and comments
                if not stripped_line or stripped_line.startswith("#"):
                    continue
                patterns.append(stripped_line)
    except Exception as e:
        logger.error(f"Failed to read .dockerignore file at {dockerignore_path}: {e}")
        return []
    return patterns


def _extract_editables_from_uv_export(project_root: Path) -> list[str]:
    """Extracts editable dependencies from a uv export output."""
    uv_export = subprocess.run(
        ["uv", "export", "--no-emit-project"], cwd=project_root, capture_output=True, text=True, check=True
    )
    matches = []
    for line in uv_export.stdout.splitlines():
        if match := re.search(r"-e\s+([^\s]+)", line):
            matches.append(match.group(1))
    return matches


def get_uv_project_editable_dependencies(project_root: Path) -> list[Path]:
    """Parses uv export output to find editable path dependencies for a given project.

    Args:
        project_root: Root of the uv project to inspect.

    Returns:
        A list of local paths referenced as editable dependencies.
    """
    paths = []
    for match in _extract_editables_from_uv_export(project_root):
        # If the the path is absolute already, keep as-is
        # otherwise we need to complete it by pre-pending the project root where 'uv export' was run from.
        resolved_path = Path(match) if Path(match).is_absolute() else (project_root / match)
        # Raise an error if the path isn't a child of the project root
        if not resolved_path.is_relative_to(project_root):
            raise ValueError(
                "Editable dependency paths must be within the project root, this is not supported."
                f"Found {resolved_path=} outside of {project_root=}."
            )
        paths.append(resolved_path)
    return paths


def get_uv_editable_install_mounts(
    project_root: Path, context_path: Path, ignore_patterns: list[str] | None = None
) -> str:
    """Builds Docker bind mounts for uv editable path dependencies.

    Args:
        project_root: Root of the uv project to inspect.
        context_path: Build context directory for Docker.
        ignore_patterns: A list of ignore patterns to apply when copying editable dependency contents.
            If None, the standard ignore patterns of 'StandardIgnore' will be used.
    Returns:
        A string of Docker bind-mount arguments for editable dependencies.
    """
    ignore_patterns = ignore_patterns or STANDARD_IGNORE_PATTERNS.copy()
    mounts = []
    for editable_dep in get_uv_project_editable_dependencies(project_root):
        # Copy the contents of the editable install by applying ignores
        editable_dep_within_context = copy_files_to_context(editable_dep, context_path, ignore_patterns=ignore_patterns)
        mounts.append(
            "--mount=type=bind,"
            f"src={editable_dep_within_context.relative_to(context_path)},"
            f"target={editable_dep.relative_to(project_root)},rw"
        )
    return " ".join(mounts)
