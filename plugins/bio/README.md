# flyteplugins-bio

Curated bioinformatics CLI tool wrappers for Flyte.

> [!WARNING]
> This plugin is still experimental and is not ready for general use or
> publication yet. The APIs, module layout, and packaging details may still
> change.

A collection of typed Flyte tasks for popular bioinformatics tools — bedtools,
samtools, bcftools, GATK, Picard, BWA, STAR, salmon, kallisto, and friends —
each shipped via its official image. Built on `flyte.extras.shell`.

Each submodule exposes:

- One or more callable shell tasks (`bedtools_intersect`, `samtools_view`, ...)
- A module-level `env` (a `flyte.TaskEnvironment`) to list in your own task
  environment's `depends_on`. Adding it registers the task and pulls/builds the
  underlying biocontainer image.

The package root also exposes `flyteplugins.bio.env`, an aggregate environment
that depends on every tool-family env — a single dependency for broad access to
the bio plugin suite.

## Install

```bash
pip install flyteplugins-bio
```

Depends only on `flyte`. The actual bio tools come from biocontainers pulled at
task execution time — nothing to install locally.

## Usage

Each wrapper is already a Flyte task, so you can run one directly:

```python
import flyte
from flyte.io import File
from flyteplugins.bio.bedtools import bedtools_sort

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(bedtools_sort.as_task(), i=File.from_local_sync("regions.bed"))
    print(run.url)
```

Each wrapper takes the tool's flags as typed keyword arguments — e.g.
`bedtools_intersect(a=..., b=[...], wa=True, f=0.5)` — and returns its
output `File`(s). To call a wrapper from inside your own task instead, add
the tool's module `env` (e.g. `from flyteplugins.bio.bedtools import env as
bedtools_env`) to your `TaskEnvironment`'s `depends_on` and `await` it.

## Running the tests

The end-to-end tests in `plugins/bio/tests/` exercise each wrapper against the
same fixtures and expected MD5s as the corresponding nf-core module. They run
**in-process by default** and submit to your configured cluster/devbox when you
pass `remote` as the first argument.

The plugin is its own uv project, so run them with `--project plugins/bio` from
the repo root (or `cd plugins/bio` first), otherwise `flyteplugins` won't be on
the path:

```bash
# one tool, locally (in-process)
uv run --project plugins/bio python plugins/bio/tests/test_bedtools.py

# one tool, on your cluster / devbox
uv run --project plugins/bio python plugins/bio/tests/test_bedtools.py remote

# the whole suite as a SINGLE run (one run tree across every tool)
uv run --project plugins/bio python plugins/bio/tests/test_all.py
uv run --project plugins/bio python plugins/bio/tests/test_all.py remote
```

Each `test_<tool>` is a `@env.task` that fetches its fixtures, calls the
wrapper, and asserts the output MD5 inline; `test_all.py` imports every tool's
`test_<tool>` and fans them out as children of one run. Fixtures are pulled from
nf-core/test-datasets on first run and cached under `tests/_fixtures/`.

## Debugging a wrapped command

Every wrapper is built with `flyte.extras.shell.create(...)`, which accepts a
`debug` flag. Set `debug=True` on the `shell.create(...)` for the tool you're
debugging (e.g. in `bedtools.py`) and re-run. The task then prints, to the
container's **real stderr** — your console locally, the task/pod logs on remote:

- the **staged input directory** (`ls -la /var/inputs`) — exactly what landed in
  the container, including the placeholders the backend writes for optional and
  scalar inputs;
- the **fully rendered command** — the exact bash after flag/template expansion;
- the tool's **captured stdout and stderr** after it runs.

This is the fastest way to see why a remote run differs from a local one — e.g.
an input that didn't stage, or a flag that expanded unexpectedly.

## Currently available

- `flyteplugins.bio.bedtools` — `bedtools_intersect`, `bedtools_sort`, `bedtools_merge`
- `flyteplugins.bio.cat` — `cat_fastq`
- `flyteplugins.bio.gunzip` — `gunzip`
- `flyteplugins.bio.untar` — `untar`

More tools are added as needed. Contributions following the same pattern (one
file per tool family, sharing one biocontainer image, exposing a module-level
`env`) are welcome once the plugin is ready to stabilize. The package-level
`flyteplugins.bio.env` is intended to grow alongside them.

## Known limitations / backlog

- **`Glob` / `Dir` outputs don't round-trip on remote yet.** A wrapper with a
  multi-file output (currently `untar`) declares a multipart `Dir` correctly —
  the serialized task interface is `blob { dimensionality: MULTIPART }` — but the
  backend's copilot uploads the directory as a single blob, so the consumer's
  `Dir` transformer rejects it (`Expected multipart, received 0`). Single-`File`
  output tools work fine on remote; `untar` only runs locally until the backend
  supports multipart raw-container outputs. `test_all` includes it, so the full
  suite is local-only for now.
