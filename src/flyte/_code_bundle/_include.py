from pathlib import Path
from typing import List

from flyte._logging import logger


class FlyteInclude:
    """
    Reads a .flyteinclude file and exposes patterns for allowlist-based bundling.

    When a .flyteinclude file is present in the bundle root, build_code_bundle()
    switches from ignore-based (walk everything, filter out) to include-based
    (only bundle what is listed here).

    Each line is a path, directory, or glob pattern relative to the root directory —
    the same syntax accepted by AppEnvironment.include and ls_relative_files():
      - A directory name (e.g. "lib1/") includes all files recursively within it.
      - A glob pattern (e.g. "src/**/*.py") is expanded via Python's glob.glob().
      - A plain file path (e.g. "config.yaml") includes that single file.

    Lines starting with # and blank lines are ignored.
    """

    INCLUDE_FILE = ".flyteinclude"

    def __init__(self, root: Path):
        self.root = root
        self.patterns = self._load_patterns()

    def _load_patterns(self) -> List[str]:
        include_file = self.root / self.INCLUDE_FILE
        if not include_file.exists():
            return []
        logger.debug(f"Found .flyteinclude at {include_file}")
        lines = include_file.read_text(encoding="utf-8").splitlines()
        patterns = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        logger.debug(f"Loaded {len(patterns)} include patterns: {patterns}")
        return patterns

    @property
    def has_patterns(self) -> bool:
        return bool(self.patterns)
