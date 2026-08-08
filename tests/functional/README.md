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

The app scenario also needs `fastapi` + `httpx` installed locally (to build the
app spec before submitting): `uv run --with fastapi --with httpx pytest …`.

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
| Per-run wait ceiling (s) | `FLYTE_FUNCTIONAL_WAIT_TIMEOUT` | `600` |
| Submit retry attempts | `FLYTE_FUNCTIONAL_SUBMIT_ATTEMPTS` | `40` |
| Submit retry delay (s) | `FLYTE_FUNCTIONAL_SUBMIT_RETRY_DELAY` | `30` |

## Layout

```
tests/functional/
├── conftest.py       # fixtures: per-test client init, marker options, ordering
├── flyte_ops.py      # shared helpers: init, submit-with-retry, assert-succeeded
├── tasks/            # one module per scenario (task/env definitions)
│   ├── simple.py  imgbuild.py  imgcache.py  reusable.py  trigger.py  app.py
└── test_*.py         # the scenarios
```

## Origin

Consolidated from several copies of the same smoke suite that had drifted apart
across different repositories. This is the shared home: decoupled from any one
CI so users can run it directly, and so other projects can reference it here
rather than maintain their own fork.
