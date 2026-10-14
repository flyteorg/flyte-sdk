"""Every flyteplugins name we tell people to install must be one we actually publish.

A distribution name that appears in a `pip install` line or in `with_pip_packages(...)` is a
name pip will resolve against PyPI. If we never upload that name it simply stays unregistered,
and because PyPI has no notion of owning the flyteplugins prefix, anyone may claim it. Whoever
does gets their build backend executed at install time wherever we told people to install it,
including inside a task image built by the remote image builder.

This has gone wrong twice: five plugin directories were missing from the publish workflow's
matrix, so their names were never uploaded, and a connector ImportError pointed at
`flyteplugins-connector`, which is not a distribution this repo has ever produced.

The publish workflow now discovers its matrix from the tree, so a plugin cannot be left
unpublished. This test covers the other half: a name referenced in an install instruction that
no package here declares.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Distributions that are first-party but published from somewhere other than this repo. Add a
# name here only after confirming it is registered on PyPI by the Flyte maintainers.
KNOWN_EXTERNAL = {
    # Connector plugin for flyte, published from its own repo.
    "flyteplugins-connectors",
    # Union SDK, the proprietary extensions for Flyte.
    "flyteplugins-union",
}

TEXT_SUFFIXES = {".py", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml"}

# A distribution name stops at an extras bracket, a version specifier or quoting.
NAME_RE = re.compile(r"flyteplugins[-_][A-Za-z0-9._-]*")
PIP_INSTALL_RE = re.compile(r"pip install[^\n]*")
# One level of nested parentheses is enough for the calls we write, e.g. a call that embeds
# f"...{flyte.version()}" as an argument.
WITH_PIP_PACKAGES_RE = re.compile(r"with_pip_packages\s*\((?:[^()]|\([^()]*\))*\)", re.DOTALL)


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.split(
        "\n"
    )
    return [REPO_ROOT / f for f in out if f and Path(f).suffix in TEXT_SUFFIXES]


def _declared_distributions() -> set[str]:
    names = set()
    for pyproject in (REPO_ROOT / "plugins").rglob("pyproject.toml"):
        if ".venv" in pyproject.parts or "site-packages" in pyproject.parts:
            continue
        match = re.search(r'^name\s*=\s*"(flyteplugins[^"]*)"', pyproject.read_text(), re.MULTILINE)
        if match:
            names.add(match.group(1))
    return names


def _normalize(name: str) -> str:
    return name.replace("_", "-").rstrip("-.")


def _referenced_names() -> dict[str, set[str]]:
    """Map each flyteplugins name used in an install instruction to the files instructing it."""
    referenced: dict[str, set[str]] = {}
    for path in _tracked_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if "flyteplugins" not in text:
            continue
        spans = [m.group(0) for m in PIP_INSTALL_RE.finditer(text)]
        spans += [m.group(0) for m in WITH_PIP_PACKAGES_RE.finditer(text)]
        for span in spans:
            for raw in NAME_RE.findall(span):
                referenced.setdefault(_normalize(raw), set()).add(str(path.relative_to(REPO_ROOT)))
    return referenced


@pytest.mark.skipif(
    subprocess.run(["git", "rev-parse"], cwd=REPO_ROOT, capture_output=True, check=False).returncode != 0,
    reason="needs a git checkout to enumerate tracked files",
)
def test_installable_plugin_names_are_published_by_this_repo():
    declared = _declared_distributions()
    assert declared, "found no flyteplugins distributions under plugins/"

    unknown = {
        name: sorted(files)
        for name, files in _referenced_names().items()
        if name not in declared and name not in KNOWN_EXTERNAL
    }

    assert not unknown, (
        "These names are installed by instructions in this repo but no package here declares "
        "them, so they are unregistered on PyPI and claimable by anyone:\n"
        + "\n".join(f"  {name}: {', '.join(files)}" for name, files in sorted(unknown.items()))
        + "\nEither add the plugin that publishes the name, correct the instruction, or list it "
        "in KNOWN_EXTERNAL if it is published from another repo."
    )
