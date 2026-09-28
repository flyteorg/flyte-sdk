"""Report the installed packages whose version differs from uv.lock, as a Markdown table.

Used by the nightly pre-release integration tests (.github/workflows/integration_tests_prerelease.yml).
That job deliberately upgrades the environment past uv.lock, so when it fails this table is the first
place to look for the upstream release that broke us.

Usage:
    uv pip freeze > resolution.txt
    python maint_tools/report_deps_vs_lock.py resolution.txt [--lock uv.lock]

Packages that are installed but absent from uv.lock (e.g. the plugins' own dependencies) have nothing
to compare against and are not listed. Always exits 0.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import tomllib
from packaging.version import InvalidVersion, Version

_NAME_SEPARATORS = re.compile(r"[-_.]+")


def _normalize(name: str) -> str:
    """PEP 503 name normalization, so `Typing_Extensions` and `typing-extensions` compare equal."""
    return _NAME_SEPARATORS.sub("-", name).lower()


def locked_versions(lock_path: Path) -> dict[str, set[str]]:
    """Map package name -> versions pinned in uv.lock (a package can be locked at several versions across forks)."""
    with lock_path.open("rb") as f:
        lock = tomllib.load(f)
    locked: dict[str, set[str]] = {}
    for package in lock.get("package", []):
        if "version" in package:
            locked.setdefault(_normalize(package["name"]), set()).add(package["version"])
    return locked


def installed_versions(freeze_lines: Iterable[str]) -> dict[str, str]:
    """Map package name -> version from `uv pip freeze` output.

    Editable (`-e ...`) and direct-URL (`name @ file://...`) installs carry no comparable version and are skipped.
    """
    installed: dict[str, str] = {}
    for raw in freeze_lines:
        line = raw.strip()
        if not line or line.startswith(("#", "-e ")) or " @ " in line:
            continue
        name, separator, version = line.partition("==")
        if separator:
            installed[_normalize(name)] = version
    return installed


def _is_prerelease(version: str) -> bool:
    try:
        return Version(version).is_prerelease
    except InvalidVersion:
        return False


def differing_packages(locked: dict[str, set[str]], installed: dict[str, str]) -> list[tuple[str, str, str, str]]:
    """Rows of (package, locked versions, installed version, pre-release marker) for every mismatch, sorted by name."""
    rows = []
    for name, version in sorted(installed.items()):
        if name in locked and version not in locked[name]:
            rows.append((name, ", ".join(sorted(locked[name])), version, "yes" if _is_prerelease(version) else ""))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("freeze", type=argparse.FileType("r"), help="`uv pip freeze` output, or '-' for stdin")
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "uv.lock",
        help="uv.lock to compare against (default: the repository's)",
    )
    args = parser.parse_args(argv)

    rows = differing_packages(locked_versions(args.lock), installed_versions(args.freeze))

    print(f"### Dependencies that differ from uv.lock ({len(rows)})")
    print()
    if not rows:
        print("Every installed package that appears in uv.lock is at its locked version.")
        return 0
    print("| Package | uv.lock | Installed | Pre-release |")
    print("| --- | --- | --- | --- |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
