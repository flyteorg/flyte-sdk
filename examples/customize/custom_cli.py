# /// script
# dependencies = [
#    "tyro",
#    "flyte>=2.0.0b20",
# ]
# ///

from dataclasses import dataclass

import tyro

import flyte

env = flyte.TaskEnvironment(
    name="custom_cli",
    image=flyte.Image.from_uv_script(__file__, name="flyte"),
)


@dataclass
class Config:
    foo: int
    bar: str = "default"


@env.task
async def main(config: Config):
    print(f"foo: {config.foo}, bar: {config.bar}")


if __name__ == "__main__":
    # Generate a CLI and instantiate `Config` with its two arguments: `foo` and `bar`.
    config = tyro.cli(Config)

    flyte.init_from_config()
    r = flyte.run(main, config)
    print(r.url)
