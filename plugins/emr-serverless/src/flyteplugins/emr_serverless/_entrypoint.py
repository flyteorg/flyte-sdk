"""Bootstrap staged to S3 for Pythonic EMR Serverless jobs.

The connector passes the serialized Flyte 2 `a0` command as this script's
arguments. `a0` performs CodeBundle loading and the standard inputs,
`outputs.pb`, and `error.pb` protocol. This shim only bridges process and EMR
status semantics.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from typing import Optional, Sequence


def _run_subprocess(command: Sequence[str]) -> int:
    """Run a command while forwarding termination to the child."""
    process = subprocess.Popen(list(command))
    signal.signal(signal.SIGTERM, lambda signum, frame: process.send_signal(signum))
    return process.wait()


def _option_value(command: Sequence[str], option: str) -> Optional[str]:
    """Return the value following a command-line option."""
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _has_runtime_error(command: Sequence[str]) -> bool:
    """Return whether `a0` wrote the standard `error.pb` artifact."""
    output_path = _option_value(command, "--outputs-path")
    if not output_path:
        return False
    from flyte import storage

    return storage.exists_sync(storage.join(output_path, "error.pb"))


def main() -> None:
    """Execute the Flyte 2 runtime command supplied by the connector."""
    command = sys.argv[1:]
    if not command or command[0] != "a0":
        print("Usage: _entrypoint.py a0 [Flyte runtime arguments...]", file=sys.stderr)
        raise SystemExit(2)
    return_code = _run_subprocess(command)
    if return_code == 0 and _has_runtime_error(command):
        print("Flyte runtime wrote error.pb; forcing EMR job failure", file=sys.stderr)
        return_code = 1
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
