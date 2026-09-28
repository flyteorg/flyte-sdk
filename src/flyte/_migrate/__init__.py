"""
Source-to-source migration of flytekit (v1) code to the flyte (v2) SDK, used by `flyte migrate`.

See `docs/design/v1-to-v2-codemod.md` for the design. Requires the `migrate` extra (`pip install "flyte[migrate]"`).
"""

from ._migrate import RULE_IDS, migrate_file, output_path_for
from ._result import MigrationError, MigrationResult, OutputExistsError, Todo
from ._writer import write_result

__all__ = [
    "RULE_IDS",
    "MigrationError",
    "MigrationResult",
    "OutputExistsError",
    "Todo",
    "migrate_file",
    "output_path_for",
    "write_result",
]
