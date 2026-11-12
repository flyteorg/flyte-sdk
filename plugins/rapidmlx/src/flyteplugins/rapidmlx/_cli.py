"""Resolve the host's Rapid-MLX command, including Homebrew outside PATH.

The Apple Silicon bridge's host environment can omit Homebrew from PATH.
This wrapper preserves server arguments and prints an install hint if missing.
"""

from __future__ import annotations

import os
import shutil
import sys

_INSTALL_HINT = """\
[flyteplugins-rapidmlx] Rapid-MLX is not installed on this machine.

Install Rapid-MLX on the Apple Silicon Mac with one of:

    brew install rapid-mlx
    pip install rapid-mlx

The pip installation requires Python 3.12+.
Then re-run `flyte serve` / redeploy the app. See the plugin README for details.
"""

# Common Homebrew prefixes for the `rapid-mlx` binary. Checked in `resolve_rapidmlx` after
# `shutil.which`, because the Apple Silicon host-execution environment's PATH
# omits Homebrew.
_HOMEBREW_RAPID_MLX_CANDIDATES = ("/opt/homebrew/bin/rapid-mlx", "/usr/local/bin/rapid-mlx")


def resolve_rapidmlx() -> str | None:
    """The path to the `rapid-mlx` binary, or None if it cannot be found anywhere."""
    found = shutil.which("rapid-mlx")
    if found:
        return found
    for candidate in _HOMEBREW_RAPID_MLX_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def main() -> int:
    binary = resolve_rapidmlx()
    if binary:
        # Hand off to the real binary, preserving argv. execv replaces this
        # process, so the real rapid-mlx becomes the long-running server and its
        # signals/log streams belong to it.
        os.execv(binary, [binary, *sys.argv[1:]])
        return 0  # not reached

    sys.stderr.write(_INSTALL_HINT)
    return 127


if __name__ == "__main__":
    sys.exit(main())
