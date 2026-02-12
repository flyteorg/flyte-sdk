"""
Example demonstrating content-based caching with custom hash methods for raw pandas DataFrames.

This example shows how to use HashFunction with typing.Annotated to enable content-based
caching for raw pandas DataFrames. The producer task outputs a pd.DataFrame with a custom
hash computed from the data via type annotations, and the consumer task uses that hash
for cache key computation.

The driver task calls the consumer twice with the same data. The first call executes the
consumer task, and the second call should hit the cache (returning the same result without
re-executing the task).

Note: Caching requires running on a remote Flyte cluster. Local execution does not support
caching.
"""

from typing import Annotated

import pandas as pd

import flyte
from flyte import Cache
from flyte.io import HashFunction

# Create task environment with required dependencies
img = flyte.Image.from_debian_base(name="pandas-dataframe-hash").with_pip_packages("pandas", "pyarrow")

env = flyte.TaskEnvironment(
    "pandas_dataframe_hash",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)

# Sample data for testing
SAMPLE_DATA = {
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "value": [100, 200, 300, 400, 500],
}


def hash_pandas_dataframe(df: pd.DataFrame) -> str:
    """
    Custom hash function for pandas DataFrames.
    Uses pandas' built-in hash_pandas_object to compute a content-based hash.
    """
    return str(pd.util.hash_pandas_object(df).sum())


# Create a reusable type alias with the hash function annotation
HashedPandasDataFrame = Annotated[pd.DataFrame, HashFunction.from_function(hash_pandas_dataframe)]


@env.task
async def produce_dataframe() -> HashedPandasDataFrame:
    """
    Producer task that creates a pandas DataFrame.

    The return type annotation includes HashFunction, which tells Flyte to compute
    a content-based hash from the DataFrame using our custom hash function.
    """
    df = pd.DataFrame(SAMPLE_DATA)
    df["action_name"] = flyte.ctx().action.run_name
    return df


@env.task(cache=Cache(behavior="override", version_override="v1"))
async def consume_dataframe(input_df: pd.DataFrame) -> str:
    """
    Consumer task that processes the pandas DataFrame.

    This task is cached based on the hash of the input DataFrame. If called with
    a DataFrame that has the same hash, the cached result will be returned.
    """
    import random

    # Generate a random number to verify cache hits
    # If the cache is hit, the same random number will be returned
    random_num = random.randint(1, 1000000)

    total_value = input_df["value"].sum()
    row_count = len(input_df)

    print(f"Processing DataFrame with {row_count} rows")
    print(f"Total value: {total_value}")

    return f"Processed {row_count} rows, total={total_value}, random={random_num}"


@env.task
async def driver() -> str:
    """
    Driver task that demonstrates cache behavior.

    Calls the producer once to create a DataFrame, then calls the consumer twice
    with the same DataFrame. The second consumer call should hit the cache.
    """
    # Produce the DataFrame with custom hash (via type annotation)
    df = await produce_dataframe()
    print(f"Produced DataFrame with {len(df)} rows")
    print(f"DataFrame hash: {hash_pandas_dataframe(df)}")

    # First consumer call - should execute the task
    print("\n=== First consumer call (should execute) ===")
    result1 = await consume_dataframe(df)
    print(f"Result 1: {result1}")

    # Second consumer call - should hit the cache
    print("\n=== Second consumer call (should hit cache) ===")
    result2 = await consume_dataframe(df)
    print(f"Result 2: {result2}")

    # Verify cache hit by checking that results are identical
    if result1 == result2:
        print("\n✓ Cache hit confirmed! Both results are identical.")
    else:
        print(f"\n✗ Cache miss! Results differ: {result1} != {result2}")

    return f"First: {result1} | Second: {result2} | Cache hit: {result1 == result2}"


if __name__ == "__main__":
    flyte.init_from_config()

    # Run the driver task
    run = flyte.run(driver)
    print(f"Run URL: {run.url}")
    run.wait()

    result = run.outputs()[0]
    print(f"\nFinal result: {result}")
