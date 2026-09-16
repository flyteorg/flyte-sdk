# Releasing `flyte` (flyte-sdk)

This document is the runbook for cutting a release of the Flyte SDK. It is the
*only* thing a maintainer needs to publish a new version. Anyone with write
access to the repo can follow it — no special release credentials beyond the
repo's own `GITHUB_TOKEN` and the standard `PYPI_USERNAME`/`PYPI_PASSWORD`
secrets are needed.

## Model

A release is a **tag + a GitHub Release**. Everything else follows from that:

1. `Tag & Release` (`.github/workflows/tag_and_release.yml`) checks that the
   version is well-formed and has a `CHANGELOG.md` entry, then tags `main` and
   opens a GitHub Release whose body is that changelog section.
2. `Publish` (`.github/workflows/publish.yml`) already listens for
   `release: published`. Creating the Release fires that event automatically, so
   the existing pipeline then builds and publishes the PyPI wheels (core
   `flyte` + every `flyteplugins-*` plugin), the RS controller wheels, the
   Docker images, the dependency constraints, and the docs-regen signal.

The two workflows are deliberately separate: this one owns the *release
ceremony* (version + changelog + tag + notes), and the other owns *distribution*
(artifacts). Keeping the ceremony in its own workflow means anyone can create a
release without touching the build pipeline, and keeps a reviewable, documented
choke point.

## Prerequisites

- Write access to `https://github.com/flyteorg/flyte-sdk`.
- A `CHANGELOG.md` whose top section reflects the upcoming release. There is no
  "release manager" role — anyone who can press "Run workflow" can do this.

## Process (checklist)

### 1. Prepare `main` for release

- [ ] Pick the version you are releasing. Tags follow `v<major>.<minor>.<patch>`
      (`v2.9.0`), optionally with a pre-release suffix `v2.9.0b1`. This must
      match the `tag_regex` in `pyproject.toml`
      (`^v(?P<version>\d+\.\d+\.\d+.*)$`) so `setuptools_scm` derives a valid
      PEP 440 wheel version.
- [ ] Make sure `main` has a `## [<version>] - TBA` section in `CHANGELOG.md`.
      If you have been adding notes under `## [Unreleased]`, move them under the
      new version heading. Keep one section per version — the workflow refuses to
      ship duplicates or empty sections:
      ```bash
      # locally verify the changelog is well-formed and has the entry
      uv run python maint_tools/update_changelog.py v2.9.0 CHANGELOG.md --only-check
      ```
      > Tip: you can regenerate a canonical, tag-derived changelog at any time
      > with `uv run python maint_tools/generate_changelog.py`. It is mainly for
      > backfilling; for normal development you append human-readable notes under
      > `## [Unreleased]` instead.
- [ ] Merge the changelog change to `main` (normal PR + review, so it crosses
      branch protection cleanly).

### 2. Trigger the release

There are two ways to kick it off — manual and programmatic. Both accept the
matching release tag and are equivalent.

**Manual (GitHub UI):**

[Run the `Tag & Release` workflow](https://github.com/flyteorg/flyte-sdk/actions/workflows/tag_and_release.yml)
and set `release_version` to `v2.9.0` (final) or `v2.9.0b1` (beta). Pick the
branch you prepared (normally `main`).

**Programmatic (from another workflow or a bot):**

Call it as a reusable workflow:

```yaml
name: Bump and release
on:
  push:
    tags: ["v*"]          # or workflow_dispatch, schedule, etc.

jobs:
  release:
    uses: flyteorg/flyte-sdk/.github/workflows/tag_and_release.yml@main
    with:
      release_version: v2.9.0
```

Or dispatch it via `repository_dispatch` (e.g. from a script or an external
system), with `release_version` in the client payload:

```bash
curl -X POST https://api.github.com/repos/flyteorg/flyte-sdk/dispatches \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -d '{"event_type":"release-request","client_payload":{"release_version":"v2.9.0"}}'
```

> **Note:** `workflow_call` is intended for callers *within this repo* (e.g. a
> bot-driven bump-and-release workflow in `flyte-sdk`). When invoked as a
> cross-repo reusable workflow, `github.ref`/`ref_name` resolve to the *caller's*
> repository, so the changelog commit and tag would target the wrong repo.
> Prefer `repository_dispatch` (or `workflow_dispatch`) for anything cross-repo.

### 3. Verify the release

- [ ] Watch the `Tag & Release` workflow. The `validate` job must pass: it
      checks the version format, that the version has a changelog entry, and
      that the tag does not already exist.
- [ ] Confirm a new Release appears at
      https://github.com/flyteorg/flyte-sdk/releases with the changelog section
      as its body. Prerelease versions (`*a*`/`*b*`) are auto-marked as
      prereleases.
- [ ] Confirm `Publish` runs (triggered by `release: published`) and completes.
      If it fails partway, fix the failure and re-run `Publish` from the Actions
      UI — publishing is gated on the release tag and is idempotent
      (`--skip-existing` on PyPI).

## What the workflow does to your repo

- Tags the release (`git tag v2.9.0`) and pushes the tag to `origin`.
- Creates (or updates) the GitHub Release with the changelog section as its body.
- Best-effort: stamps `CHANGELOG.md` with today's date
  (`## [v2.9.0] - 2026-01-31`) and commits that to the branch. On repos with
  strict branch protection this push may be refused — that's handled gracefully
  and does not block the release. Prefer merging the date-stamped changelog via
  PR when you can.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `validate` fails: *"version 'v2.9' does not match"* | The version is malformed. Use `v<major>.<minor>.<patch>` (e.g. `v2.9.0`). |
| `validate` fails: *"v2.9.0 is not in CHANGELOG.md"* | No `## [v2.9.0]` section exists. Add/move the notes and re-run. |
| `validate` fails: *"tag v2.9.0 already exists"* | That version is already tagged. Delete the tag, cut a new version, or — if you're repairing a release that half-finished — re-run with `allow_existing_tag=true` (refreshes the existing GitHub Release; it pushes no code or tag). |
| `update_changelog.py` reports *"Duplicate changelog heading(s)"* | Two `## [v2.9.0]` headings exist (often after a rebase). Merge them by hand. |
| Release created but `Publish` did not run | Check the workflow run's `release: published` event — creating a Release auto-fires it. Ensure the release isn't left as a draft. |
| PyPI upload rejected a *local* version (`2.3.1.dev0+g…`) | The working tree was dirty so `setuptools_scm` invented a version. Re-run `Publish` from the clean release tag (it pins the tag version). |

## Versioning

Use [SemVer](https://semver.org). The current convention on the repo has been:
- `v2.<minor>.<patch>` — stable SDK releases.
- `v2.<minor>.<patch>b<N>` — betas leading up to a minor, published to PyPI as
  pre-releases.

`setuptools_scm` derives the wheel version from the tag, so the tag *is* the
version — pick it deliberately.
