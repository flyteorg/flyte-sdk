import logging

import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    image=flyte.Image.from_debian_base(name="vscode"),
    resources=flyte.Resources(cpu=1.5, memory="1500Mi"),
)


@env.task
async def say_hello(data: str) -> str:
    print(f"Hello, world! - {flyte.ctx().action}")
    return f"Hello {data}"


@env.task
async def say_hello_nested(data: str = "default string") -> str:
    return await say_hello(data=data)


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml", log_level=logging.DEBUG)
    run = flyte.with_runcontext(env_vars={"LOG_LEVEL": "10", "_F_E_VS": "True"}).run(
        say_hello_nested, data="hello world", n=10
    )
    print(run.name)
    print(run.url)
