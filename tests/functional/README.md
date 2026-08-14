# Functional (environment-validation) tests

A small, self-contained suite that runs the core flyte v2 scenarios end-to-end
against a **real Flyte backend** to show a deployment is healthy. Point it at
your environment and it answers: *can I run tasks, get their I/O and logs, build
and cache images, run reusable actors, serve apps, and register triggers?*

It's meant as the shared "prove my setup works" suite — the same scenarios that
today live copied across cloud, unionai-docs, and helm-charts, consolidated here.

## Scenarios

| Test | Marker | Checks |
| --- | --- | --- |
| `test_simple` | | Submit the basic task and wait for a successful terminal state |
| `test_image_builder` | | Build a custom image (`requests`) on the backend and run a task on it |
| `test_image_cache` | | Run the same stable-image task twice; the second run hits the image cache |
| `test_io` | | A completed run's `outputs` are retrievable |
| `test_logs` | `logs` | The run's logs are retrievable through the SDK (best-effort — WARNs, never fails) |
| `test_trigger` | | Deploy a scheduled trigger, verify its automation spec, toggle it inactive |
| `test_reusable` | * | Fan out work over a `ReusePolicy` (actor) environment |
| `test_app` | `app` | Deploy a FastAPI app, hit `/` and `/health`, then deactivate |

Every scenario is self-contained — it submits its own run and waits — so there
is no shared fixture coupling and the suite parallelises cleanly (see *Running*).
Scenarios run lightest-first (image builds warm the backend) and the app last.

\* `test_reusable`, `test_trigger`, and `test_app` exercise Union-platform
features (actor reuse, triggers, app serving). On a backend that doesn't support
one, skip it (see below).

## Running

These are marked `integration`, so the default unit run
(`pytest -k "not integration"`) skips them. To run them you need a **configured
backend**. Two ways to point the suite at one:

**1. Your standard Flyte config** (the common path) — with `~/.flyte/config.yaml`
or `FLYTE_CONFIG` set, just run:

```bash
uv run pytest tests/functional -m integration -v
```

**2. Explicit endpoint** (CI / ad-hoc), via env vars:

```bash
export FLYTE_FUNCTIONAL_ENDPOINT="my-org.my-flyte.example.com"
export FLYTE_FUNCTIONAL_API_KEY="…"      # optional; else uses config auth (Pkce)
export FLYTE_FUNCTIONAL_ORG="my-org"     # optional
export FLYTE_FUNCTIONAL_PROJECT="…"      # optional; else config default
uv run pytest tests/functional -m integration -v
```

**Standalone** (outside flyte-sdk — another project's CI, or just this directory) —
the suite's runtime dependencies are declared in `tests/functional/pyproject.toml`
(`flyte`, `fastapi`/`httpx` for the app scenario, `pytest` + xdist/rerunfailures),
so install and run against that:

```bash
uv run --project tests/functional pytest tests/functional -m integration -v
```

`flyte` is unpinned there so you validate against a current SDK — pin it on your
side if your backend is a release behind (see *Backend flavour & retry tuning*).

### Options

- `--skip <scenario>` — skip any scenario a backend doesn't support, by marker,
  test name, or short name (repeatable or comma-separated). E.g. `--skip app` (no
  app serving), `--skip logs` (no queryable log store), `--skip app,logs`, or
  `--skip reusable --skip trigger`.
- Transient infra blips (image-pull, propagation lag) are worth one retry:
  `--reruns 1 --reruns-delay 15` (needs `pytest-rerunfailures`).
- **Parallelism.** The scenarios are independent, so run them concurrently with
  `pytest-xdist`: `-n auto` (one worker per CPU) collapses wall-clock toward the
  slowest single scenario instead of the sum. On a small/single-node backend,
  cap it (e.g. `-n 2`) so concurrent image builds don't overwhelm the builder.

### Configuration reference

All env vars are optional. A CI fallback name is accepted for each so an existing
pipeline can reuse this suite with minimal change.

| Purpose | Preferred var | CI fallback |
| --- | --- | --- |
| Endpoint | `FLYTE_FUNCTIONAL_ENDPOINT` | `CONTROL_PLANE_URL` |
| API key | `FLYTE_FUNCTIONAL_API_KEY` | `FLYTE_API_KEY` |
| Org | `FLYTE_FUNCTIONAL_ORG` | `ORG_NAME` |
| Project | `FLYTE_FUNCTIONAL_PROJECT` | `CLUSTER_NAME` |
| Domain | `FLYTE_FUNCTIONAL_DOMAIN` | (`development`) |
| Env-name suffix | `FLYTE_FUNCTIONAL_SUFFIX` | `ENV_SUFFIX` |
| Queue pin (optional) | `FLYTE_FUNCTIONAL_QUEUE` | `CLUSTER_NAME` |

### Backend flavour & retry tuning

The retry logic is portable across backends. `FLYTE_FUNCTIONAL_BACKEND` (`oss`, the
default, or `union`) selects which submission errors count as transient — Union's
managed data plane flaps differently (tunnel/proxy, cluster-pool routing) than an
OSS `flyteadmin` + propeller. Set it to `union` against a managed data plane; the
OSS-specific transient set is an empty placeholder for now (common blips still
retry). Everything below is optional with a sensible default:

| Purpose | Var | Default |
| --- | --- | --- |
| Backend flavour (transient-error set) | `FLYTE_FUNCTIONAL_BACKEND` | `oss` |
| Extra transient substrings (comma-sep) | `FLYTE_FUNCTIONAL_TRANSIENT_MARKERS` | (none) |
| Image-build cache bust (ephemeral store) | `FLYTE_FUNCTIONAL_IMAGE_CACHE_BUST` | (none) |
| Install the suite into scenario images (no-checkout consumers) | `FLYTE_FUNCTIONAL_SUITE_SPEC` | (none) |
| Per-run wait ceiling (s) | `FLYTE_FUNCTIONAL_WAIT_TIMEOUT` | `600` |
| Submit retry attempts | `FLYTE_FUNCTIONAL_SUBMIT_ATTEMPTS` | `40` |
| Submit retry delay (s) | `FLYTE_FUNCTIONAL_SUBMIT_RETRY_DELAY` | `30` |

## Layout

```
tests/functional/
├── pyproject.toml            # installable dist: flyte-functional-tests (+ [app]/[harness] extras)
├── README.md
└── flyte_functional_tests/   # the importable package
    ├── plugin.py     # pytest11 plugin: --skip option, markers, lightest-first ordering
    ├── conftest.py   # fixtures: per-test client init; eager task registration
    ├── flyte_ops.py  # shared helpers: init, submit-with-retry, assert-succeeded
    ├── tasks/        # one module per scenario (task/env definitions)
    │   ├── simple.py  imgbuild.py  imgcache.py  reusable.py  trigger.py  app.py
    └── test_*.py     # the scenarios
```

## Consuming from another repo (cloud, flyte-agent-plugins, …)

The suite is a real installable package (`flyte-functional-tests`), so another repo's CI
doesn't vendor or path-hack it — it installs it and runs `pytest --pyargs`. Each **scenario's
task/app pod** still needs the task source, which flyte delivers via its fast-register code
bundle (`loaded_modules` — the imported modules whose `__file__` is under the run's working
dir; site-packages is excluded). So install the suite **editable from a checkout that stays
under the working dir**, and run from `tests/functional/` so `root_dir == cwd` makes
`flyte_functional_tests` the bundle root (the pods re-import the modules by that name):

```bash
# Check out the suite (a released tag once published, or a branch ref for pre-merge), then:
uv pip install --prerelease=allow "flyte==2.5.20"                 # pin flyte for backend compat
uv pip install --prerelease=allow -e "flyte-sdk-suite/tests/functional[app,harness]"
cd flyte-sdk-suite/tests/functional
pytest --pyargs flyte_functional_tests -m integration -v         # --skip app,trigger,reusable on an OSS backend
```

`flyte` 2.x is currently a pre-release on PyPI, so installs need `--prerelease=allow`. Extras:
`[app]` = fastapi/httpx (the app scenario); `[harness]` = flyteplugins-union (only if you drive
cluster pool/queue/routing ops from the same env).

**Experimental — fully decoupled, no checkout (`FLYTE_FUNCTIONAL_SUITE_SPEC`).** The intent is
to install the package (version or git ref) with no source in the working dir and set
`FLYTE_FUNCTIONAL_SUITE_SPEC` so each scenario *image* installs the suite (pods import from
site-packages). This does **not work yet**, blocked on two flyte-core gaps: (1) a run with no
in-tree source raises `CodeBundleError` ("no files to bundle") — flyte needs a "code is in the
image, skip the bundle" signal; and (2) the remote image builder can't `pip install` the suite
from a git+pre-release spec (`ImageBuildError`) — it doesn't pass `--prerelease`. The hook is
implemented and ready for when those land; until then use the editable-from-checkout mode above.

## Origin

Consolidated from several copies of the same smoke suite that had drifted apart
across different repositories. This is the shared home: decoupled from any one
CI so users can run it directly, and so other projects can reference it here
rather than maintain their own fork.
