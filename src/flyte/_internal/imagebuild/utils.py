import shutil
from pathlib import Path


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
    :param ignore_patterns: List of file and folder patterns to ignore when copying folders
    """
    if src.is_absolute() or ".." in str(src):
        dst_path = context_path / str(src.absolute()).replace("/", "./_flyte_abs_context/", 1)
    else:
        dst_path = context_path / src
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        # TODO: Add support dockerignore
        default_ignore_patterns = [".idea", ".venv"]
        ignore_patterns = list(set(ignore_patterns + default_ignore_patterns))
        shutil.copytree(src, dst_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*ignore_patterns))
    else:
        shutil.copy(src, dst_path)
    return dst_path
