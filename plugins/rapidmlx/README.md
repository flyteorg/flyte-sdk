# Flyte Rapid-MLX Plugin

Serve LLMs locally on Apple Silicon using [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX)
and Flyte Apps. Rapid-MLX is an open-source local LLM inference server built on MLX
with an OpenAI-compatible API.

`RapidMLXAppEnvironment` defaults to `gpu="AppleSilicon"`. On a devbox started with
`--apple-silicon`, the bridge runs `rapid-mlx serve` natively on the Mac and forwards
the app's URL, port and logs. This plugin requires an Apple Silicon Mac; the server
runs on the host rather than inside the Linux devbox container.

## Installation

Install Rapid-MLX on the Mac:

```bash
brew install rapid-mlx
# Or, with Python 3.12+:
pip install rapid-mlx
```

Install the Flyte plugin from this checkout before publication:

```bash
uv pip install -e plugins/rapidmlx
```

Verify the server installation with `rapid-mlx --help`.
The default image installs only the Flyte plugin, which must be available to the
host environment's package installer when deploying the app. A release of
`flyteplugins-rapidmlx` is needed for the default image to work without a custom image.

## Quickstart

`examples/accelerators/rapidmlx_serve.py` defines:

```python
from flyteplugins.rapidmlx import RapidMLXAppEnvironment

rapidmlx_app = RapidMLXAppEnvironment(
    name="rapidmlx-local",
    model_id="qwen3.5-4b-4bit",
)
```

Run it against an Apple Silicon devbox:

```bash
flyte start devbox --apple-silicon
flyte create config --devbox
python examples/accelerators/rapidmlx_serve.py
```

The example prints the forwarded app URL. Append `/v1` to that URL for an OpenAI
client. For a standalone server started with `rapid-mlx serve qwen3.5-4b-4bit`, the
usual base URL is `http://localhost:8000/v1`:

```python
from openai import OpenAI

# Replace with the forwarded app URL + /v1 when using the devbox.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
model = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Use `/v1/models` to obtain the actual served model id; it may differ from the CLI
alias. No API key is required by default; a placeholder works for clients that
require one. With auth enabled, send the configured key.

| Endpoint | Description |
|----------|-------------|
| `/v1/models` | List served model ids; linked from the Flyte app |
| `/v1/chat/completions` | OpenAI chat completions |
| `/v1/responses` | OpenAI Responses |
| `/v1/embeddings` | Embeddings when an embedding model is served |
| `/v1/messages` | Anthropic Messages (base URL without `/v1`) |

## Configuration

The plugin constructs `rapid-mlx serve <model_id> --host 127.0.0.1 --port <port>`.

| Field | Description |
|-------|-------------|
| `name` | App name (required, lowercase and hyphens). |
| `model_id` | Positional MLX model id or alias. Default `qwen3.5-4b-4bit`. |
| `port` | Explicit bind port, default `8000`. Choose another if occupied. |
| `resources` | Defaults to `gpu="AppleSilicon"`; other GPUs are rejected. |
| `api_key` | Optional `--api-key`. Prefer secrets with `env_vars` for real keys. |
| `env_vars` | Server environment, including optional `RAPID_MLX_API_KEY`. |
| `extra_args` | List or shell-split string appended to the command. See `rapid-mlx serve --help`. |
| `image` | Image whose pip packages are rebuilt for the Mac's host environment. |
| `scaling` | `flyte.app.Scaling` configuration. |

Rapid-MLX selects the first free port in 8000–8009 when no port is specified.
This plugin always passes an explicit port so Flyte's forwarding stays aligned;
an occupied port fails without fallback. Another local server on port 8000 requires
a different `port` value.

Models use the standard Hugging Face cache (`~/.cache/huggingface/hub`, honoring
`HF_HOME` and `HF_HUB_CACHE`). No separate model directory is needed.

## How it works

The devbox bridge runs `fserve` on the Mac in an environment built from the image's
pip packages. `fserve` launches the plugin's `flyte-rapidmlx` wrapper, which replaces
itself with the host's `rapid-mlx` command, preserving arguments and signals.
The wrapper searches PATH, then `/opt/homebrew/bin` and `/usr/local/bin`, since the
bridge's host environment can omit Homebrew from PATH. If the executable is missing,
it prints installation instructions and exits with code 127.

The server must already be installed on the Mac. The plugin does not install
Rapid-MLX into the Linux image and does not connect to an existing server; serving
an app starts its own server process on the Mac.
