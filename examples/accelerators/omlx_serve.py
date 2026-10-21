"""
Serve an LLM on the Mac's GPU with oMLX from an app on an Apple Silicon devbox.

[oMLX](https://github.com/jundot/omlx) is an OpenAI-compatible inference server built on Apple's
MLX framework. oMLX is macOS + Apple Silicon only, and a devbox container is a Linux VM with no
Metal, so this `OMLXAppEnvironment` requests `gpu="AppleSilicon"` (the plugin defaults to it): the
devbox's bridge runs `omlx serve` natively on the Mac, in an environment built from the image's
pip packages, and forwards the app's URL, port and logs.

This runs workload code outside any sandbox, so it is opt-in, and the devbox binds every port to
127.0.0.1:

    flyte start devbox --apple-silicon
    flyte create config --devbox
    python examples/accelerators/omlx_serve.py

From another shell, the app answers the OpenAI API (the example prints its URL):

    curl http://omlx-local-<project>-<domain>.localhost:30081/v1/models

    python - <<'PY'
    from openai import OpenAI
    client = OpenAI(base_url="http://omlx-local-<project>-<domain>.localhost:30081/v1", api_key="not-used")
    resp = client.chat.completions.create(
        model="qwen3-0.6b",  # each subdir of oMLX's model dir is a served id
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(resp.choices[0].message.content)
    PY

Prerequisites on the Mac:

- oMLX installed: `brew tap jundot/omlx https://github.com/jundot/omlx && brew install omlx`
  (or `pip install git+https://github.com/jundot/omlx`). Models live under `~/.omlx/models` by
  default; oMLX downloads an MLX model from HuggingFace on first use.
- `uv` on the Mac: the bridge builds each workload's environment with it.
- The flyteplugins-omlx plugin installed (or on `PYTHONPATH`) for this example.

Setting `FLYTE_APPLE_SILICON_EXECUTION=container` switches to a host-side MLX daemon instead, but
oMLX always runs its own `omlx serve` process on the Mac.
"""

from flyteplugins.omlx import OMLXAppEnvironment

import flyte

omlx_app = OMLXAppEnvironment(
    name="omlx-local",
    # oMLX serves whatever is under ~/.omlx/models by default. Set model_dir to a different
    # (expanded, absolute) folder. model_id is a friendly id surfaced in the Flyte UI and used as
    # the OpenAI `model`; oMLX itself serves each subdirectory of the model dir as a model.
    model_id="qwen3-0.6b",
    # Conservative defaults; tune for your machine. See `omlx serve --help`.
    extra_args=[
        "--max-concurrent-requests",
        "4",
        "--log-level",
        "info",
    ],
    # gpu="AppleSilicon" is the plugin default; the app runs natively on the Mac's GPU.
)

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(omlx_app)
    print(f"oMLX app is up at {app.url} — try {app.url}/v1/models")
    print("Press Ctrl+C to stop.")
    try:
        import signal

        signal.pause()
    except (KeyboardInterrupt, AttributeError):
        pass
