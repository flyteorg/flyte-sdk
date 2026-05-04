"""
A simple oMLX app example for serving an MLX model locally on Apple Silicon.

[oMLX](https://github.com/jundot/omlx) is an OpenAI-compatible inference
server built on Apple's MLX framework. This example shows how to wrap it in
a Flyte AppEnvironment and serve it on your laptop.

Prerequisites
-------------

You need a Mac with Apple Silicon (M1/M2/M3/M4) and oMLX installed::

    pip install omlx
    pip install --pre flyteplugins-omlx

Drop one or more MLX models into ``~/.omlx/models``. The simplest way is to
let oMLX fetch one on first use — for example point the OpenAI client at the
HuggingFace repo id ``mlx-community/Llama-3.2-1B-Instruct-4bit``.

Run locally
-----------

::

    flyte serve --local examples/genai/omlx/omlx_app.py omlx_app

You should see the app start on ``http://localhost:8000``. Then::

    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
    resp = client.chat.completions.create(
        model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        messages=[{"role": "user", "content": "Hello, MLX!"}],
    )
    print(resp.choices[0].message.content)
"""

from flyteplugins.omlx import OMLXAppEnvironment

import flyte.app

omlx_app = OMLXAppEnvironment(
    name="omlx-local",
    # oMLX defaults to ~/.omlx/models when --model-dir is omitted. Set this
    # explicitly if your models live elsewhere.
    model_dir="",
    # Friendly id surfaced in the Flyte UI. The actual model name in OpenAI
    # client requests is whichever subdir / HF repo id oMLX exposes.
    model_id="mlx-default",
    port=8000,
    # Conservative defaults; tune for your machine. See `omlx serve --help`.
    extra_args=[
        "--max-concurrent-requests",
        "4",
        "--log-level",
        "info",
    ],
    # Local serve mode is single-replica by definition.
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
    requires_auth=False,
)


if __name__ == "__main__":
    # When run directly, serve locally without going through `flyte serve`.
    import flyte

    handle = flyte.with_servecontext(mode="local").serve(omlx_app)
    handle.activate(wait=True)
    print(f"oMLX app is up at {handle.endpoint} — try {handle.endpoint}/v1/models")
    print("Press Ctrl+C to stop.")
    try:
        import signal

        signal.pause()
    except (KeyboardInterrupt, AttributeError):
        pass
    finally:
        handle.deactivate(wait=True)
