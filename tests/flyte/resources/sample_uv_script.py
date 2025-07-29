# /// script
# dependencies = [
#    "polars",
#    "httpx",
#    "numpy",
# ]
# ///

import flyte

env = flyte.TaskEnvironment(
    name="hello_world",
    image=flyte.Image.from_uv_script(script=__file__, name="hello_world"),
)


@env.task
async def create_dataframe():
    return {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 32, 37],
        "city": ["New York", "Paris", "Berlin"],
    }


if __name__ == "__main__":
    pass
