"""
Helpers for resolving `Environment.include` entries into absolute paths suitable
for the code bundler.

Relative paths are anchored at the directory of each environment's
`_declaring_file` (the user file where the env was instantiated), matching the
convention established by `AppEnvironment`. Absolute paths pass through
unchanged. Directories and glob patterns are deferred to `ls_relative_files`
downstream — this module only resolves strings into absolute form.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from flyte._environment import Environment


def collect_env_include_files(envs: Iterable[Environment]) -> Tuple[str, ...]:
    """
    Resolve every env's ``include`` entries to absolute paths.

    - Relative paths anchor at ``Path(env._declaring_file).parent``.
    - Absolute paths pass through as-is.
    - Deduplicated while preserving first-seen order.
    - Raises ``ValueError`` if an env has a non-empty ``include`` but no
      ``_declaring_file`` (shouldn't happen in normal usage).
    """
    resolved: list[str] = []
    seen: set[str] = set()
    for env in envs:
        if not env.include:
            continue
        if env._declaring_file is None:
            raise ValueError(
                f"Environment {env.name!r} declares include={env.include!r} but its "
                f"declaring file could not be determined. This usually indicates the env "
                f"was constructed from an unusual context (e.g., a REPL with no file)."
            )
        anchor = Path(env._declaring_file).parent
        for inc in env.include:
            p = Path(inc)
            abs_path = str(p if p.is_absolute() else (anchor / p).resolve())
            if abs_path not in seen:
                seen.add(abs_path)
                resolved.append(abs_path)
    return tuple(resolved)
