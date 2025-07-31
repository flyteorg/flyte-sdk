import flyte

QUICK_THINKING_LLM = "gpt-4o-mini"
DEEP_THINKING_LLM = "o4-mini"

env = flyte.TaskEnvironment(
    name="trading-agents",
    secrets=[
        flyte.Secret(key="finnhub_api_key", as_env_var="FINNHUB_API_KEY"),
        flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY"),
    ],
    image=flyte.Image.from_uv_script("main.py", name="trading-agents", pre=True),
    resources=flyte.Resources(cpu="1"),
)
