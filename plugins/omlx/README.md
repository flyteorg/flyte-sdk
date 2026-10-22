# Flyte oMLX Plugin

Serve LLMs locally on Apple Silicon using [oMLX](https://github.com/jundot/omlx)
and Flyte Apps.

This plugin provides the `OMLXAppEnvironment` class for defining an oMLX-backed
app environment. oMLX is an OpenAI-compatible inference server built on Apple's
MLX framework, macOS + Apple Silicon only.

The plugin integrates with the **Apple Silicon flyte devbox**: an
`OMLXAppEnvironment` requests `gpu="AppleSilicon"` by default, so the devbox's
bridge runs `omlx serve` natively on the Mac, in an environment built from the
image's pip packages, and forwards the app's URL, port and logs.

> **Mac-only.** oMLX runs only on macOS 15+ with Apple Silicon (M1/M2/M3/M4).
> `mlx` has no Linux wheels, so this plugin cannot run in a Linux container or on
> a cloud cluster. It is designed for a devbox started with `--apple-silicon`.

## Installation

oMLX is **not on PyPI** — install it via Homebrew, the macOS `.dmg`, or from
source. See the upstream
[install guide](https://github.com/jundot/omlx#installation):

```bash
# oMLX itself (one of):
brew tap jundot/omlx https://github.com/jundot/omlx && brew install omlx
# or:
pip install git+https://github.com/jundot/omlx

# Then the Flyte plugin:
pip install --pre flyteplugins-omlx
```

Verify the install with `omlx --help` before running the plugin.

## Quickstart — serve an LLM from the Apple Silicon devbox

`examples/accelerators/omlx_serve.py`:

```python
import flyte
from flyteplugins.omlx import OMLXAppEnvironment

omlx_app = OMLXAppEnvironment(
    name="omlx-local",
    # oMLX serves whatever is under ~/.omlx/models by default. Set model_dir to a
    # different folder (an absolute, expanded path).
    model_id="qwen3-0.6b",
    extra_args="--log-level info",
)
```

Run it against a devbox started with the Apple Silicon bridge:

```bash
flyte start devbox --apple-silicon
flyte create config --devbox
python examples/accelerators/omlx_serve.py
```

`OMLXAppEnvironment` defaults to `gpu="AppleSilicon"`, so the app runs natively
on the Mac's GPU. From another shell, hit the OpenAI-compatible API (the app's
URL is printed by the example):

```python
from openai import OpenAI

client = OpenAI(base_url="http://omlx-local-...:30081/v1", api_key="not-used")
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

## Configuration

`OMLXAppEnvironment` constructs the `omlx serve --host 127.0.0.1 --port <port> ...`
invocation that the devbox runs on the Mac. Recognised fields:

| Field         | Description                                                                                       |
|---------------|---------------------------------------------------------------------------------------------------|
| `name`        | App name (required, lowercase + hyphens).                                                         |
| `port`        | Bind port. Default `8000` (oMLX default).                                                         |
| `model_dir`   | Local directory containing MLX-format model subdirectories, mapped to `--model-dir`. Use an expanded absolute path. |
| `model_id`    | Friendly id surfaced in Flyte / OpenAI client requests; oMLX serves each subdir of `model_dir`.   |
| `resources`   | `gpu="AppleSilicon"` by default (runs on the Mac via the devbox bridge). Other GPUs are rejected. |
| `api_key`     | Optional `--api-key` for the oMLX server. For real secrets, prefer `secrets` + `env_vars`.        |
| `extra_args`  | List or string of extra flags passed to `omlx serve`. See `omlx serve --help`.                    |
| `env_vars`    | Environment variables for the server process (e.g. `HF_TOKEN`).                                   |
| `image`       | Container image. Sets the pip packages rebuilt into the Mac's host environment.                   |
| `scaling`     | `flyte.app.Scaling` config.                                                                       |

### Example: tighten memory / cache config

```python
omlx_app = OMLXAppEnvironment(
    name="omlx-local",
    model_dir="/Users/me/.omlx/models",
    extra_args=(
        "--max-model-memory 24GB "
        "--max-concurrent-requests 4 "
        "--paged-ssd-cache-dir /Users/me/.omlx/ssd_cache "
        "--paged-ssd-cache-max-size 50GB "
        "--log-level info"
    ),
)
```

## How it works

With `flyte start devbox --apple-silicon`, a `gpu="AppleSilicon"` app runs
**natively on the Mac**. The devbox container (a Linux VM) has no Metal, so the
bridge hands the app's entrypoint (`fserve`) to the Mac, which runs it in a `uv`
environment built from the image's pip packages. `fserve` then runs the app's
`args` — here `omlx serve ...` — on the Mac, and the devbox forwards the app's
URL, port and logs.

The `args` invoke a tiny `flyte-omlx` console-script wrapper (installed by this
package) which `execv`s into the real `omlx` binary. Because the Apple Silicon
host environment's PATH only contains its own `bin/` (Homebrew is not on it), the
wrapper also probes `/opt/homebrew/bin` and `/usr/local/bin` so a
Homebrew-installed oMLX is found. If oMLX isn't installed, the wrapper prints an
actionable install hint instead of a raw `FileNotFoundError`.

## Limitations

- **Apple Silicon only.** oMLX uses `mlx`, which only ships for darwin/arm64. You
  cannot run this plugin in a Linux container today.
- oMLX must be installed on the Mac (Homebrew or pip-from-git). The default image
  installs only the plugin, because `mlx` has no Linux wheels with which to build a
  base image.
