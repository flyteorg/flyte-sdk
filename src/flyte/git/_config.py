import pathlib
import subprocess


def config_from_root() -> pathlib.Path:
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get git root directory: {result.stderr}")
    root = pathlib.Path(result.stdout.strip())
    return root / ".flyte" / "config.yaml"
