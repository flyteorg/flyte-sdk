"""Serve an LLM on an Apple Silicon devbox with Rapid-MLX.

Install Rapid-MLX on the Mac with `brew install rapid-mlx` or
`pip install rapid-mlx` (Python 3.12+), and install the Flyte plugin.
The devbox bridge also needs `uv` on the Mac.

```bash
flyte start devbox --apple-silicon
flyte create config --devbox
python examples/accelerators/rapidmlx_serve.py
```

The app runs natively on the Mac's GPU and prints its forwarded URL. Append
`/v1` to that URL for an OpenAI client, and use `/v1/models` to discover the
served model id. Standalone Rapid-MLX uses `http://localhost:8000/v1` by default.
No API key is required by default (`api_key="not-needed"` works with OpenAI clients).

Models are downloaded into the standard Hugging Face cache. This app starts
its own server; choose another `port` if 8000 is occupied.
"""

from flyteplugins.rapidmlx import RapidMLXAppEnvironment

import flyte

rapidmlx_app = RapidMLXAppEnvironment(
    name="rapidmlx-local",
    model_id="qwen3.5-4b-4bit",
    # gpu="AppleSilicon" is the plugin default.
)

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(rapidmlx_app)
    print(f"Rapid-MLX app is up at {app.url} — try {app.url}/v1/models")
    print("Press Ctrl+C to stop.")
    try:
        import signal

        signal.pause()
    except (KeyboardInterrupt, AttributeError):
        pass
