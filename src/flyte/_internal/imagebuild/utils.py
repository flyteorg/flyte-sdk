import shutil
from pathlib import Path
from typing import List, Optional

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
        dst_path = context_path / str(src.absolute()).replace("/", "./_flyte_abs_context/", 1)
    else:
        dst_path = context_path / src
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        default_ignore_patterns = [".idea", ".venv"]
        ignore_patterns = list(set(ignore_patterns + default_ignore_patterns))
        shutil.copytree(src, dst_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*ignore_patterns))
    else:
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
