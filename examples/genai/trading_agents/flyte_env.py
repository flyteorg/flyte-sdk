import flyte

QUICK_THINKING_LLM = "gpt-4o-mini"
DEEP_THINKING_LLM = "o4-mini"
REGISTRY = "ghcr.io/flyteorg"

env = flyte.TaskEnvironment(
    name="trading-agents",
    secrets=[
        flyte.Secret(key="finnhub_api_key", as_env_var="FINNHUB_API_KEY"),
        flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY"),
    ],
    image=flyte.Image.from_uv_script(
        "main.py",
        name="trading-agents",
        platform=("linux/amd64", "linux/arm64"),
        pre=True,
        registry=REGISTRY,
    ),
    resources=flyte.Resources(cpu="1"),
    cache="disable",
    # TODO: Enable after the reuse policy is enabled on all tenants
    # reusable=flyte.ReusePolicy(replicas=20, idle_ttl=10),
)
