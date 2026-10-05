"""Tasks you cannot edit: an ordinary wrapper task carries the declaration (proposal section 8).

`legacy_clean` is deployed by another team and declares nothing. The adapter wraps the remote reference
and declares, in its own decorator, what the wrapped task really produces. It is a second producer of
`events` (allowed: partitions keep producers distinct), so it is deployed separately and is not part of the
default e2e walk; see the README.

    flyte deploy --root-dir . legacy/legacy_clean.py env
    flyte deploy --root-dir . legacy/adapter.py env
"""

from __future__ import annotations

from datetime import datetime

from ingest.events import events, raw_events

import flyte
import flyte.remote
from flyte.io import DataFrame, File

env = flyte.TaskEnvironment(name="ingest-adapter", image=flyte.Image.from_debian_base(), labels={"team": "ml"})

legacy_clean = flyte.remote.Task.get("ingest-legacy.legacy_clean", auto_version="latest")


@env.task(
    consumes_artifacts={
        "raw": raw_events,
        "date": raw_events.get_partition_value("date"),
        "region": raw_events.get_partition_value("region"),
    },
    produces_artifacts=(events,),
)
async def clean_adapter(raw: File, date: datetime, region: str) -> DataFrame:
    return await legacy_clean(raw=raw, min_quality=30)
