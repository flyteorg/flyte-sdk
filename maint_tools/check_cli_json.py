#!/usr/bin/env python3
"""Check that `flyte gen docs --type json` describes the real command tree.

Reads the JSON on stdin. Run by `make cli-docs-gen`, which is the only thing
that exercises the doc generator against the live CLI rather than against
commands a test constructed.

That distinction is the point. The unit tests build their own click objects, so
they cannot see the failures that only exist on the real tree: a walker that
stopped descending because a private type it matches by class NAME was renamed,
or a genuine command whose default does not serialise. Both produce a generator
that exits 0 and emits a plausible, wrong document.

The floor is deliberately far below the real count. It is here to catch a
collapse, not to be updated every time a command is added.
"""

import json
import sys

MIN_COMMANDS = 20
REQUIRED_KEYS = {"path", "name", "is_group", "distribution", "help", "arguments", "options"}


def main() -> None:
    try:
        doc = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        sys.exit(f"ERROR: `flyte gen docs --type json` did not emit valid JSON: {exc}")

    commands = doc.get("commands")
    if not isinstance(commands, list):
        sys.exit("ERROR: no `commands` list in the output.")

    if len(commands) < MIN_COMMANDS:
        sys.exit(
            f"ERROR: only {len(commands)} command(s) described, below the floor of {MIN_COMMANDS}.\n"
            "       The generator exited 0, so the likely cause is a tree walk that stopped\n"
            "       descending -- check whether a private type `walk_commands` matches on by\n"
            "       class name was renamed."
        )

    for command in commands:
        missing = REQUIRED_KEYS - command.keys()
        if missing:
            sys.exit(f"ERROR: {command.get('path', '<unnamed>')} is missing {sorted(missing)}.")

    if not any(c["path"] == "flyte" for c in commands):
        sys.exit("ERROR: the root command is absent, so the walk did not start where it should.")

    print(f"check-cli-json: {len(commands)} commands described, all keys present")


if __name__ == "__main__":
    main()
