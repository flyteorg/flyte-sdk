"""`llama-cpp-fserve`: resolve mounted GGUF weights, then exec llama-server.

The app environment mounts model weights as a *directory* (the GGUF filename inside a
`RunOutput`/blob-store directory is unknown at deploy time), but llama-server takes a
path to a concrete `.gguf` file. This shim bridges the two: it rewrites

- `--model-dir <dir>` -> `--model <resolved .gguf>`
- `--draft-model-dir <dir>` -> `--model-draft <resolved .gguf>`

leaving every other argument untouched, and then replaces itself with llama-server.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import sys
from pathlib import Path

from flyteplugins.llamacpp._constants import LLAMA_SERVER_BINARY

logger = logging.getLogger(__name__)

_DIR_FLAG_TO_MODEL_FLAG = {
    "--model-dir": "--model",
    "--draft-model-dir": "--model-draft",
}


def find_gguf(path: str) -> str:
    """Resolve the GGUF file to serve from a mounted file or directory.

    GGUFs directly in `path` win over any in subdirectories: a `--model-dir` may point at a
    mount that also holds the draft/MTP GGUF in a *subdirectory* (e.g. an object-store FUSE
    prefix carrying both `Model.gguf` and `MTP/draft.gguf`), and a recursive match could
    otherwise resolve the model to the draft. Only if no GGUF sits directly in `path` do we
    recurse. For sharded models only the first shard is passed to llama-server (it discovers
    the rest itself), so `*-00001-of-*.gguf` wins over other matches.
    """
    if os.path.isfile(path):
        return path
    matches = sorted(glob.glob(os.path.join(path, "*.gguf")))
    if not matches:
        matches = sorted(glob.glob(os.path.join(path, "**", "*.gguf"), recursive=True))
    if not matches:
        raise FileNotFoundError(f"No .gguf files found under {path!r}")
    first_shards = [m for m in matches if "-00001-of-" in Path(m).name]
    return first_shards[0] if first_shards else matches[0]


def build_command(argv: list[str]) -> list[str]:
    """Build the llama-server argv, resolving `--model-dir`/`--draft-model-dir`."""
    server = shutil.which("llama-server") or LLAMA_SERVER_BINARY
    cmd = [server]
    i = 0
    while i < len(argv):
        arg = argv[i]
        model_flag = _DIR_FLAG_TO_MODEL_FLAG.get(arg)
        if model_flag is not None:
            if i + 1 >= len(argv):
                raise ValueError(f"{arg} requires a value")
            cmd.extend([model_flag, find_gguf(argv[i + 1])])
            i += 2
        else:
            cmd.append(arg)
            i += 1
    return cmd


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cmd = build_command(sys.argv[1:])
    logger.info("Starting llama-server: %s", " ".join(cmd))
    os.execv(cmd[0], cmd)
