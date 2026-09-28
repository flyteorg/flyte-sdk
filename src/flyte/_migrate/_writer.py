from __future__ import annotations

import os
import tempfile

from ._result import MigrationResult, OutputExistsError


def write_result(result: MigrationResult, *, force: bool = False) -> None:
    """Atomically write the migrated code next to the source file. The source file is never modified."""
    if result.status == "nothing_to_migrate":
        return
    output = result.output_path
    if output.exists() and not force:
        raise OutputExistsError(f"{output} already exists. Pass --force to overwrite it.")
    fd, tmp = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(result.code)
        # mkstemp creates the file as 0600; give the output the same permissions as its source.
        os.chmod(tmp, result.source_path.stat().st_mode & 0o777)
        os.replace(tmp, output)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
