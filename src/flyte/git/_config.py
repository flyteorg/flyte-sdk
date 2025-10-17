import pathlib
import subprocess

import flyte.config


def config_from_root(path: pathlib.Path | str = ".flyte/config.yaml") -> flyte.config.Config | None:
    """Get the config file from the git root directory.

    By default, the config file is expected to be in `.flyte/config.yaml` in the git root directory.
    """
    try:
        result = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=False, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        root = pathlib.Path(result.stdout.strip())
        if not (root / path).exists():
            return None
        return flyte.config.auto(root / path)
    except Exception:
        return None
