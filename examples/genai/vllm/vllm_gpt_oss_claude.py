"""
Serve openai/gpt-oss-20b with vLLM and connect Claude Code to it.

vLLM exposes an OpenAI-compatible API (``/v1/chat/completions``), but Claude
Code speaks the Anthropic API (``/v1/messages``). To bridge them we run
``claude-code-router`` (``ccr``) locally — it translates Anthropic ↔ OpenAI
and injects the ``ANTHROPIC_*`` env vars before launching ``claude``.

Deploy
------

```
flyte deploy examples/genai/vllm/vllm_gpt_oss_claude.py vllm_app
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
      "models": ["gpt-oss-20b"]
    }
  ],
  "Router": {
    "default": "vllm,gpt-oss-20b",
    "background": "vllm,gpt-oss-20b",
    "think": "vllm,gpt-oss-20b",
    "longContext": "vllm,gpt-oss-20b",
    "longContextThreshold": 120000,
    "webSearch": "vllm,gpt-oss-20b"
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
/model vllm,gpt-oss-20b
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

from flyteplugins.vllm import DEFAULT_VLLM_IMAGE, VLLMAppEnvironment

import flyte.app

vllm_app = VLLMAppEnvironment(
    name="gpt-oss-claude-code",
    model_hf_path="openai/gpt-oss-20b",
    model_id="gpt-oss-20b",
    resources=flyte.Resources(cpu="6", memory="24Gi", gpu="A10G:1", disk="200Gi"),
    # The plugin's default image, renamed so this app's builds are cached separately from
    # the other vLLM examples'. Nothing extra is needed to serve gpt-oss.
    image=DEFAULT_VLLM_IMAGE.clone(name="vllm-claude-image"),
    stream_model=True,
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=3600,
    ),
    requires_auth=False,
    extra_args=[
        "--served-model-name gpt-oss-20b",
        "--enable-auto-tool-choice",
        "--tool-call-parser openai",
        "--max-model-len 131072",
        "--gpu-memory-utilization 0.92",
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
