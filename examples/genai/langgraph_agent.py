# /// script
# requires-python = "==3.13"
# dependencies = [
#     "langgraph==0.6.4",
#     "langchain==0.3.27",
#     "langchain-anthropic==0.3.18",
#     "flyte==0.1.0",
# ]
# ///


from langchain_core.messages import BaseMessage
from langgraph.prebuilt import create_react_agent

import flyte

agent_env = flyte.TaskEnvironment(
    name="agent",
    secrets=[
        flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
    ],
    image=flyte.Image.from_uv_script(script=__file__, name="langgraph-agent", pre=True),
    resources=flyte.Resources(cpu=1),
)


def get_weather_tool(city: str) -> str:
    """Get weather for a given city."""
    return get_weather(city)


@agent_env.task
def get_weather(city: str) -> str:
    return f"It's always sunny in {city}!"


@agent_env.task
def main(in_str: str) -> list[BaseMessage]:
    # Create a React agent with the weather tool
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest", tools=[get_weather_tool], prompt="You are a helpful assistant"
    )

    # Run the agent
    print("Running agent...")
    output = agent.invoke({"messages": [{"role": "user", "content": in_str}]})

    return output["messages"]


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")
    r = flyte.run(main, in_str="what is the weather in sf")
    print(r.url)
