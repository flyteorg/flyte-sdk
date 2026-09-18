"""Retroactively build ``CHANGELOG.md`` from the git history.

Generator used once to backfill every release the repo has already cut, and safe
to re-run any time to regenerate a canonical, tag-derived changelog. It walks the
release tags in chronological order and, for each consecutive pair, collects the
commits that landed between them (``git log <previous>..<tag>``), classifies each
by its conventional-commit type, and renders a Markdown section per release.

Sections are written newest-first, like the repo convention. The indentation and
heading style match what the rest of the release tooling expects (``## [version]``
headings, one per version).

Only tags that look like ``v<major>.<minor>.<patch>`` are treated as releases;
non-release tags (``backup/*``, ``rs-v0.1.0``) are ignored.

Usage:
    python maint_tools/generate_changelog.py [--output CHANGELOG.md]
"""

import argparse
import re
import subprocess
from datetime import date
from pathlib import Path

RELEASE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)([ab]\d+)?$")


def sh(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _sort_key(tag: str, when: date) -> tuple:
    """Order (date, version) so same-day tags sort in version order.

    Ordering by tag date alone (with the stable refname tiebreak) is fragile: if
    betas like ``v2.0.0b9`` and ``v2.0.0b10`` were tagged on the same day, the
    alphabetical fallback ``b10 < b9`` would put them out of version order and
    the ``previous..tag`` diff below would attribute the wrong commits. Tie-break
    with a numeric version tuple so this is always correct.
    """
    m = RELEASE_TAG_RE.match(tag)
    major, minor, patch = (int(m.group(i)) for i in (1, 2, 3))
    pre = m.group(4) or ""
    if pre:
        pre_key = (0, ord(pre[0]), int(pre[1:]))  # alpha/beta before final
    else:
        pre_key = (1, 0, 0)  # final release sorts after its prereleases
    return (when.year, when.month, when.day, major, minor, patch, *pre_key)


def release_tags() -> "list[tuple[str, date]]":
    """Release tags in chronological order as (tag, tagger_date)."""
    refs = sh("for-each-ref", "--format=%(refname:short) %(creatordate:short)", "refs/tags")
    tags = []
    for line in refs.splitlines():
        tag, _, when = line.partition(" ")
        if not RELEASE_TAG_RE.match(tag):
            continue
        try:
            d = date.fromisoformat(when)
        except ValueError:  # unparsable/missing date -> fall back to nobody
            d = date.fromisoformat("1970-01-01")
        tags.append((tag, d))
    # Chronological (oldest first) for tag-to-tag diffing.
    tags.sort(key=lambda t: _sort_key(t[0], t[1]))
    return tags


def commits_between(previous: str, tag: str) -> "list[str]":
    """Non-merge commit subjects reachable from ``tag`` but not ``previous``."""
    if previous is None:
        out = sh("log", "--no-merges", "--format=%s", tag, "--")
    else:
        out = sh("log", "--no-merges", "--format=%s", f"{previous}..{tag}", "--")
    return [s for s in out.splitlines() if s and "Prepare changelog" not in s]


def strip_subject(subject: str) -> str:
    """Drop a conventional-commit type prefix/scope so a line reads as English.

    ``feat(hermes): durable turns (#1591)`` -> ``durable turns (#1591)``,
    otherwise the subject is returned unchanged.
    """
    m = re.match(r"^([a-z]+)(?:\([^)]*\))?:\s*(.*)$", subject)
    return m.group(2).strip() if m else subject


def render() -> str:
    tags = release_tags()
    lines: "list[str]" = []
    lines.append("# Changelog")
    lines.append("")
    lines.append("All notable changes to this project are documented in this file.")
    lines.append("")
    lines.append("## [Unreleased]")
    lines.append("")
    lines.append(
        "_Add changelog entries here as PRs merge; they move under a `## [vX.Y.Z] - TBA` heading at release time._"
    )
    lines.append("")

    sections = []  # list of (tag, when, [bullets])
    previous = None
    for tag, when in tags:
        commits = commits_between(previous, tag)
        sections.append((tag, when, commits))
        previous = tag

    # Newest first.
    for tag, when, bullets in reversed(sections):
        lines.append(f"## [{tag}] - {when.isoformat()}")
        lines.append("")
        if bullets:
            for bullet in bullets:
                lines.append(f"- {strip_subject(bullet)}")
        else:
            lines.append("- No changelog entries.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("CHANGELOG.md"))
    args = parser.parse_args()

    count = len(release_tags())
    rendered = render()
    args.output.write_text(rendered)
    print(f"Wrote {count} release sections to {args.output}")
