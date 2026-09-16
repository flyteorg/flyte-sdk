"""Inspect and update ``CHANGELOG.md`` during a release.

This is the tool the ``Tag & Release`` workflow drives. It does four things:

* ``validate_changelog`` — reject a structurally broken changelog (duplicate or
  empty ``## [version]`` sections) before it can ship bad release notes.
* ``--only-check`` — verify that the version being released already has a
  ``## [vX.Y.Z]`` section (run before wasting a full CI run).
* date stamping — replace a ``## [vX.Y.Z] - TBA`` heading with the current date.
* ``--release_changelog`` — write just the body of the released section so the
  workflow can feed it to ``gh release create -F``.

Sections follow the format ``## [version]`` or ``## [version] - 2026-01-31``.
Keep one section per version, exactly once.
"""

import re
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


VERSION_HEADING_RE = re.compile(r"^## \[([\w\.]+)\]", re.MULTILINE)


def parse_sections(changelog_content: str) -> "list[tuple[str, str]]":
    """Every ``## [version]`` section, in file order, as (version, body).

    A list rather than a dict: two headings for the same version are a real
    (and silent) failure mode, and a dict would collapse them.
    """
    matches = list(VERSION_HEADING_RE.finditer(changelog_content))
    starts = [m.start() for m in matches]
    splits = [[start, end] for start, end in zip(starts, [*starts[1:], None])]

    def _strip_first_line(content: str) -> str:
        lines = content.splitlines()
        return "\n".join(lines[1:]).strip()

    return [
        (match.group(1), _strip_first_line(changelog_content[start:end].strip()))
        for match, (start, end) in zip(matches, splits)
    ]


def validate_changelog(changelog_content: str) -> None:
    """Reject a structurally broken changelog before it reaches a release."""
    sections = parse_sections(changelog_content)

    seen: dict = {}
    duplicates = []
    for version, _ in sections:
        if version in seen and version not in duplicates:
            duplicates.append(version)
        seen[version] = True
    if duplicates:
        fail(
            "Duplicate changelog heading(s) for: "
            + ", ".join(f"## [{v}]" for v in duplicates)
            + ".\nEach version must appear exactly once. This usually means a new "
            "section was prepended above a heading the release workflow had "
            "already stamped with a date; merge the two by hand."
        )

    empty = [v for v, body in sections if not body.strip()]
    if empty:
        fail(
            "Changelog section(s) with no content: "
            + ", ".join(f"## [{v}]" for v in empty)
            + ".\nAn empty section is a leftover heading, not a release note."
        )


def update_date_in_changelog(changelog_content: str, version: str) -> str:
    """Stamp ``## [version] - <today>`` onto the matching heading (in place)."""
    current_time = datetime.now(timezone.utc)
    lines = changelog_content.splitlines()

    output_lines = []
    for line in lines:
        if line.startswith(f"## [{version}]"):
            date_str = current_time.strftime("%Y-%m-%d")
            output_lines.append(f"## [{version}] - {date_str}")
        else:
            output_lines.append(line)

    output_lines.append("")
    return "\n".join(output_lines)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Version being released, e.g. v2.9.0")
    parser.add_argument("changelog_file", type=Path)
    parser.add_argument("--release_changelog", type=Path)
    parser.add_argument(
        "--only-check",
        action="store_true",
        help="Only check that the version exists in the changelog; do not stamp the date",
    )

    args = parser.parse_args()

    assert args.changelog_file.exists(), f"missing changelog file: {args.changelog_file}"

    changelog_content = args.changelog_file.read_text()

    validate_changelog(changelog_content)
    sections = dict(parse_sections(changelog_content))

    full_version = args.version
    if not full_version.startswith("v"):
        fail(f"{full_version} must start with 'v' (e.g. v2.9.0)")
    if full_version not in sections:
        fail(
            f"{full_version} is not in {args.changelog_file}!"
            " Add a '## [{full_version}] - TBA' section (move items under it from"
            " '## [Unreleased]') before releasing."
        )

    if args.only_check:
        print(f"Check successful: {full_version} is in the changelog")
        sys.exit(0)

    new_changelog = update_date_in_changelog(changelog_content, full_version)
    args.changelog_file.write_text(new_changelog)

    if args.release_changelog:
        args.release_changelog.write_text(sections[full_version])
        print(f"Wrote release notes for {full_version} to {args.release_changelog}")

    print(f"Stamped {full_version} with today's date in {args.changelog_file}")
