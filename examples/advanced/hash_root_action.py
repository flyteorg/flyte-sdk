"""
Content-based caching for the *root* task of a run.

`hash_flyte_dataframe.py` shows content-based caching between tasks: a driver produces a
DataFrame and calls a cached consumer twice, and the second call hits. This example covers the
other entrypoint — passing a locally-built DataFrame straight into `flyte.run(...)`, so the
cached task *is* the root action of the run.

The two paths compute the cache key differently. A sub-action's key is computed by the
controller in-process, which substitutes `Literal.hash` for the literal's contents. A root
action's key is derived from the offloaded inputs, which reference the upload URI — and every
`flyte.run` uploads the DataFrame to a fresh URI. So without a content hash the second run
misses even though the bytes are identical.

Passing `hash_method=` to `DataFrame.from_local_sync` makes both runs agree: the key follows
the content, not where it happened to land in blob storage.

Run it twice; `check_cache_hit` below asserts the second run returns the first run's value.
"""

import pandas as pd

import flyte
from flyte import Cache
from flyte.io import DataFrame, HashFunction

img = flyte.Image.from_debian_base(name="flyte-root-hash").with_pip_packages("pandas", "pyarrow")

env = flyte.TaskEnvironment(
    "flyte_root_action_hash",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)

SAMPLE_DATA = {"id": [1, 2, 3, 4, 5], "value": [100, 200, 300, 400, 500]}


def hash_pandas_dataframe(df: pd.DataFrame) -> str:
    """Content-based hash: the same rows always produce the same digest."""
    return str(pd.util.hash_pandas_object(df).sum())


@env.task(cache=Cache(behavior="override", version_override="v1"))
async def main(df: DataFrame) -> str:
    """Cached root task.

    The random number is the cache probe: it is regenerated on every real execution, so two
    runs returning the same string can only mean the second one was served from the cache.
    """
    import random

    pdf = await df.open(pd.DataFrame).all()
    return f"rows={len(pdf)}, total={pdf['value'].sum()}, random={random.randint(1, 1000000)}"


def build_input() -> DataFrame:
    """The DataFrame to submit, tagged with a content-based hash.

    Without `hash_method` the cache key would follow the (per-run, always new) upload URI and
    the second run would miss.
    """
    return DataFrame.from_local_sync(
        pd.DataFrame(SAMPLE_DATA),
        hash_method=HashFunction.from_fn(hash_pandas_dataframe),
    )


if __name__ == "__main__":
    flyte.init_from_config()

    # Two independent submissions of the same content. Each uploads to its own URI.
    run1 = flyte.run(main, df=build_input())
    print(f"Run 1: {run1.url}")
    run1.wait()
    result1 = run1.outputs()[0]

    run2 = flyte.run(main, df=build_input())
    print(f"Run 2: {run2.url}")
    run2.wait()
    result2 = run2.outputs()[0]

    print(f"\nRun 1: {result1}")
    print(f"Run 2: {result2}")
    if result1 == result2:
        print("\n✓ Cache hit — the new upload URI did not change the cache key.")
    else:
        print("\n✗ Cache miss — the root action's key still tracks the upload URI.")
