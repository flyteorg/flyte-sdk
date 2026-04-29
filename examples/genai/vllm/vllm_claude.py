"""
Serve a model with vLLM and connect Claude Code to it.

vLLM exposes an OpenAI-compatible API (``/v1/chat/completions``), but Claude
Code speaks the Anthropic API (``/v1/messages``). To bridge them we run
``claude-code-router`` (``ccr``) locally — it translates Anthropic ↔ OpenAI
and injects the ``ANTHROPIC_*`` env vars before launching ``claude``.

Deploy
------

```
flyte deploy examples/genai/vllm/vllm_claude.py vllm_app
```

The deploy prints the app URL, e.g.
``https://<app-name>.apps.<your-domain>``. Use this URL below.

Install the router
------------------

```bash
npm install -g @musistudio/claude-code-router
```

Configure the router
--------------------

Create ``~/.claude-code-router/config.json``:

```json
{
  "Providers": [
    {
      "name": "vllm",
      "api_base_url": "https://<app-name>.apps.<your-domain>/v1/chat/completions",
      "api_key": "dummy",
      "models": ["my-model"]
    }
  ],
  "Router": {
    "default": "vllm,my-model",
    "background": "vllm,my-model",
    "think": "vllm,my-model",
    "longContext": "vllm,my-model",
    "longContextThreshold": 120000,
    "webSearch": "vllm,my-model"
  }
}
```

Connect Claude Code
-------------------

```bash
# Make sure no ANTHROPIC_* env vars are set — ccr injects its own.
unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN \
      ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL \
      ANTHROPIC_DEFAULT_HAIKU_MODEL

ccr code
```

``ccr code`` starts a local proxy on ``127.0.0.1:3456``, points Claude Code
at it, and forwards traffic to the vLLM app. Use ``ccr stop`` / ``ccr start``
to manage the proxy; logs live at ``~/.claude-code-router/claude-code-router.log``.

Optionally add ``"CLAUDE_CODE_ATTRIBUTION_HEADER": "0"`` to
``~/.claude/settings.json`` to preserve prefix caching performance.

Notes
-----

* ``--max-model-len 131072`` matches Claude Code's large system prompt; smaller
  values cause ``max_tokens must be at least 1`` errors when the prompt
  approaches the context limit.
* ``gpt-oss-20b`` ships in MXFP4 quantization which requires GPU compute
  capability ≥ 8.0 (Ampere+). T4 (sm_7.5) is not supported; L4, L40S, A10G,
  A100, and H100 all work.
"""

from flyteplugins.vllm import VLLMAppEnvironment

import flyte.app

vllm_app = VLLMAppEnvironment(
    name="gpt-oss-claude-code",
    model_hf_path="openai/gpt-oss-20b",
    model_id="my-model",
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu="L40s:1", disk="200Gi"),
    image=(
        flyte.Image.from_debian_base(
            name="vllm-claude-image",
            install_flyte=False,
        )
        .with_pip_packages("flashinfer-python", "flashinfer-cubin")
        .with_pip_packages("flashinfer-jit-cache", index_url="https://flashinfer.ai/whl/cu129")
        .with_pip_packages("vllm==0.11.0", "transformers==4.57.6")
        .with_pip_packages("flyteplugins-vllm")
    ),
    stream_model=True,
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=60,
    ),
    requires_auth=False,
    extra_args=[
        "--served-model-name my-model",
        "--enable-auto-tool-choice",
        "--tool-call-parser openai",
        "--max-model-len 131072",
        "--gpu-memory-utilization 0.92"
    ],
)


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.remote import Run

    flyte.init_from_config()

    run: Run = flyte.prefetch.hf_model(repo="openai/gpt-oss-20b", force=1)
    run.wait()
    print(run.url)

    app = flyte.serve(
        vllm_app.clone_with(
            name=vllm_app.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run.name),
            model_hf_path=None,
        )
    )
    print(f"Deployed vLLM app: {app.url}")
