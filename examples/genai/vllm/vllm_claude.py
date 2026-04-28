"""
Serve a model with vLLM and connect Claude Code to it.

Based on the vLLM Claude Code integration guide:
https://docs.vllm.ai/en/stable/serving/integrations/claude_code/

Deploy
------

```
flyte deploy examples/genai/vllm/vllm_claude.py vllm_app
```

Connect Claude Code
-------------------

Once deployed, point Claude Code at the app endpoint:

```bash
ANTHROPIC_BASE_URL=<your-app-endpoint> \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_OPUS_MODEL=my-model \
ANTHROPIC_DEFAULT_SONNET_MODEL=my-model \
ANTHROPIC_DEFAULT_HAIKU_MODEL=my-model \
claude
```

Optionally add `"CLAUDE_CODE_ATTRIBUTION_HEADER": "0"` to `~/.claude/settings.json`
to preserve prefix caching performance.
"""

from flyteplugins.vllm import VLLMAppEnvironment

import flyte
import flyte.app

vllm_app = VLLMAppEnvironment(
    name="gpt-oss-claude-code",
    model_hf_path="openai/gpt-oss-120b",
    model_id="my-model",
    resources=flyte.Resources(cpu="16", memory="200Gi", gpu="H100:4", disk="500Gi"),
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
        scaledown_after=300,
    ),
    requires_auth=False,
    extra_args=[
        "--served-model-name my-model",
        "--enable-auto-tool-choice",
        "--tool-call-parser openai",
    ],
)


if __name__ == "__main__":
    import flyte.prefetch
    from flyte.remote import Run

    flyte.init_from_config()

    run: Run = flyte.prefetch.hf_model(repo="openai/gpt-oss-120b")
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
