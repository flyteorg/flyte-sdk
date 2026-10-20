"""A task somebody else owns and that declares nothing (stands in for code you cannot edit).

flyte deploy --root-dir . legacy/legacy_clean.py env
"""

from __future__ import annotations

import io

import flyte
from flyte.io import DataFrame, File

env = flyte.TaskEnvironment(
    name="ingest-legacy",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"),
)


@env.task
async def legacy_clean(raw: File, min_quality: int = 30) -> DataFrame:
    import pandas as pd

    async with raw.open("rb") as fh:
        df = pd.read_csv(io.BytesIO(bytes(await fh.read())))
    return DataFrame.from_df(df[df["quality"] >= min_quality].reset_index(drop=True))
