# Flyte oMLX Plugin

Serve LLMs locally on Apple Silicon using [oMLX](https://github.com/jundot/omlx)
and Flyte Apps.

This plugin provides the `OMLXAppEnvironment` class for defining an oMLX-backed
app environment that can be served via `flyte serve --local` on macOS, or
embedded in larger Flyte workflows.

> **Mac-only.** oMLX runs only on macOS 15+ with Apple Silicon (M1/M2/M3/M4).
> The plugin's primary use case is local serving on a developer's Mac. Remote
> deployment to Linux nodes is not supported by oMLX itself.

## Installation

oMLX is **not on PyPI** — install it via Homebrew, the macOS `.dmg`, or from
source. See the upstream
[install guide](https://github.com/jundot/omlx#installation):

```bash
# Pick one for oMLX itself:
brew tap jundot/omlx https://github.com/jundot/omlx && brew install omlx
# or:
pip install git+https://github.com/jundot/omlx

# Then the Flyte plugin:
pip install --pre flyteplugins-omlx
```

Verify the install with `omlx --help` before running the plugin.

## Quickstart — local serving

`examples/genai/omlx/omlx_app.py`:

```python
import flyte
import flyte.app
from flyteplugins.omlx import OMLXAppEnvironment

omlx_app = OMLXAppEnvironment(
    name="omlx-local",
    # Each subdirectory of model_dir becomes a served model_id. Leave unset
    # to use the oMLX default of ~/.omlx/models.
    model_dir="~/.omlx/models",
    model_id="qwen3-0.6b",
    port=8000,
    extra_args=["--max-concurrent-requests", "4"],
)
```

Run it:

```bash
flyte serve --local examples/genai/omlx/omlx_app.py omlx_app
```

You'll see something like:

```
App 'omlx-local' is being served locally
➡️  http://localhost:8000
```

The server exposes the standard OpenAI-compatible API. From another shell:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
resp = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

Useful endpoints (also surfaced as Flyte app links):

| Endpoint           | Description                                  |
|--------------------|----------------------------------------------|
| `/v1/chat/completions` | OpenAI chat completions                  |
| `/v1/completions`      | OpenAI completions                       |
| `/v1/embeddings`       | OpenAI embeddings                        |
| `/v1/models`           | List available models                    |
| `/health`              | Health check (used by Flyte's local probe) |

## Configuration

`OMLXAppEnvironment` is a thin wrapper that constructs the
`omlx serve --host 0.0.0.0 --port <port> ...` invocation. Recognised fields:

| Field         | Description                                                                                       |
|---------------|---------------------------------------------------------------------------------------------------|
| `name`        | App name (required, lowercase + hyphens).                                                         |
| `port`        | Bind port. Default `8000` (oMLX default).                                                         |
| `model_dir`   | Local directory containing MLX-format model subdirectories. Mapped to `--model-dir`.              |
| `model_id`    | Friendly id; surfaced in Flyte but not passed to oMLX (oMLX uses subdir names / HF repo ids).     |
| `api_key`     | Optional `--api-key` for the oMLX server. For real secrets, prefer `secrets` + `env_vars`.        |
| `extra_args`  | List or string of extra flags passed to `omlx serve`. See `omlx serve --help` for the full list.  |
| `env_vars`    | Environment variables for the server process (e.g. `HF_TOKEN`).                                   |
| `image`       | Container image — placeholder; only matters if you attempt remote deploy.                         |
| `scaling`     | `flyte.app.Scaling` config (remote deploy only).                                                  |

### Example: tighten memory / cache config

```python
omlx_app = OMLXAppEnvironment(
    name="omlx-local",
    model_dir="~/.omlx/models",
    extra_args=(
        "--max-model-memory 24GB "
        "--max-concurrent-requests 4 "
        "--paged-ssd-cache-dir ~/.omlx/ssd_cache "
        "--paged-ssd-cache-max-size 50GB "
        "--log-level info"
    ),
)
```

## How it works

`flyte serve --local` runs the app's `args` as a subprocess on your local
machine and probes `http://localhost:<port>/health` until it responds. There is
no Docker, no remote scheduling, and no model streaming — oMLX loads weights
directly from disk (or downloads them from HuggingFace on first use).

The `args` invoke a tiny `flyte-omlx` console-script wrapper (installed by
this package) which `execv`s into the real `omlx` binary. If `omlx` isn't
installed, the wrapper prints an actionable install hint instead of a raw
`FileNotFoundError`.

That's why the plugin is intentionally small: it does not need the
remote-streaming model loader shim that the vLLM and SGLang plugins use.

## Limitations

- **Apple Silicon only.** oMLX uses `mlx`, which only ships for darwin/arm64.
  You cannot run this plugin in a Linux container today.
- The default `image` field exists for API symmetry with other app
  environments, but a remote deploy will fail to install `mlx`. Treat this
  plugin as local-first.
