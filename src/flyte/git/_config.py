import pathlib
import subprocess

import flyte.config


def config_from_root(path: pathlib.Path | str = ".flyte/config.yaml") -> flyte.config.Config:
    """Get the config file from the git root directory.

    By default, the config file is expected to be in `.flyte/config.yaml` in the git root directory.
    """

    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get git root directory: {result.stderr}")
    root = pathlib.Path(result.stdout.strip())
    return flyte.config.auto(root / path)
