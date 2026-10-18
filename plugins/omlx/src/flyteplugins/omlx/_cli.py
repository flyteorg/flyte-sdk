"""Friendly preflight wrapper around the upstream `omlx` console command.

An `OMLXAppEnvironment` serves via `omlx serve`. The local-serve / Apple Silicon
host process invokes `flyte-omlx` instead of `omlx` directly so that, if oMLX isn't
installed on the Mac, the user sees a clear install hint instead of a raw
`FileNotFoundError: 'omlx'`.

When oMLX *is* installed, this wrapper is transparent: it `execv`'s into the real
binary (or imports `omlx.cli` if the package is installed but the console script
isn't on PATH for some reason).

`gpu="AppleSilicon"` workloads on the flyte devbox run their app natively on the Mac
in a `uv` environment whose `PATH` only contains that environment's bin directory. A
Homebrew-installed oMLX lives at `/opt/homebrew/bin` (Apple Silicon) or
`/usr/local/bin` (Intel Macs and older setups), neither of which is on that `PATH`, so
`shutil.which("omlx")` alone would miss it. We probe those locations explicitly.
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

Then re-run `flyte serve` / redeploy the app. See the plugin README for details.
"""

# Common Homebrew prefixes for the `omlx` binary. Checked in `resolve_omlx` after
# `shutil.which`, because the Apple Silicon host-execution environment's PATH
# omits Homebrew.
_HOMEBREW_OMLX_CANDIDATES = ("/opt/homebrew/bin/omlx", "/usr/local/bin/omlx")


def resolve_omlx() -> str | None:
    """The path to the `omlx` binary, or None if it cannot be found anywhere."""
    found = shutil.which("omlx")
    if found:
        return found
    for candidate in _HOMEBREW_OMLX_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def main() -> int:
    binary = resolve_omlx()
    if binary:
        # Hand off to the real binary, preserving argv. execv replaces this
        # process, so the real omlx becomes the long-running server and its
        # signals/log streams belong to it.
        os.execv(binary, [binary, *sys.argv[1:]])
        return 0  # not reached

    # Fall back to the installed Python package, in case the script wasn't
    # placed on PATH (e.g. an unusual install layout).
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
