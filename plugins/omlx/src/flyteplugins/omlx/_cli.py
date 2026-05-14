"""Friendly preflight wrapper around the upstream `omlx` console script.

The local-serve subprocess invokes `flyte-omlx` instead of `omlx` so that, if
oMLX isn't installed on the host, the user sees a clear install hint instead
of a raw ``FileNotFoundError: 'omlx'``.

When oMLX *is* installed, this wrapper is transparent: it ``execv``'s into
the real binary (or imports ``omlx.cli`` if the package is installed but the
console script isn't on PATH for some reason).
"""

from __future__ import annotations

import os
import shutil
import sys

_INSTALL_HINT = """\
[flyteplugins-omlx] oMLX is not installed on this machine.

oMLX is macOS + Apple Silicon only and is not published to PyPI. Install it
with one of:

    brew tap jundot/omlx https://github.com/jundot/omlx && brew install omlx
    pip install git+https://github.com/jundot/omlx

Or download the .dmg from https://github.com/jundot/omlx/releases.

Then re-run `flyte serve --local`. See the plugin README for details.
"""


def main() -> int:
    omlx_bin = shutil.which("omlx")
    if omlx_bin:
        # Hand off to the real binary, preserving argv. execv replaces this
        # process so the real omlx becomes the long-running server.
        os.execv(omlx_bin, [omlx_bin, *sys.argv[1:]])
        return 0  # not reached

    # Fall back to the installed Python package, in case the script wasn't
    # placed on PATH (e.g. unusual install layouts).
    try:
        import omlx.cli as omlx_cli  # type: ignore[import-not-found]
    except ImportError:
        sys.stderr.write(_INSTALL_HINT)
        return 127

    sys.argv = ["omlx", *sys.argv[1:]]
    rc = omlx_cli.main()
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
