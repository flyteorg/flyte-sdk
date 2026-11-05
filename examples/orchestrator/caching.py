"""
Cached tasks called from a regular task and from an orchestrator
================================================================

A task's cache key is its name, interface, cache version and inputs. Who
calls it does not count: a ``flyte.orchestrator.task``, which a sandbox
leaseworker runs without a pod, launches a child with the same cache key as
a regular ``@env.task`` making the same call. So the two drivers here share
one cache. Nothing has to be deployed first::

    python caching.py

That runs ``regular_driver`` and then ``orchestrator_driver`` with the same
inputs. Both make the same six calls, in the same order. In the first run
the first call to each task misses the cache and takes a few seconds; the
rest hit. In the second run every call hits, including those the
orchestrator makes, which shows its children reuse what a regular task
cached. ``--orchestrator-first`` runs them the other way round, so the
regular task hits what the orchestrator cached.

Both drivers call each task the same ways:

- the same call twice in a row: the second hits the cache.
- ``matrix_total`` again with only ``note`` changed: ``note`` is left out of
  its cache key with ``ignored_inputs``, so this hits too.
- ``label`` with ``limit=None`` and with ``limit=2``: different inputs, so
  the second is a miss of its own the first time round.

The script prints each run's child actions with their cache status. The UI
shows the same: open a run, select a child action, and its cache status is
shown as a hit or a miss.

The inputs include a seed, the current time unless ``--seed`` is given, so
each invocation starts from a cold cache. Run it twice with the same
``--seed`` and every call in both runs hits.

``--local`` runs both drivers in this process. The local cache only serves
tasks with ``cache="auto"``, so there ``label``, whose version is set by
hand, misses every time; on a cluster it hits like the others.

The values crossing between the orchestrator and its tasks are plain data:
None, bool, int, float, str, files and directories, and lists and
string-keyed dicts of them, nested. Dataframes, Pydantic models and
dataclasses can pass through an orchestrator but it cannot read them, so they
are not used here. The tasks
are called by keyword, which works for local and remote tasks alike.
"""

import argparse
import time
from typing import Optional

import flyte
import flyte.orchestrator
import flyte.remote

env = flyte.TaskEnvironment(name="orchestrator-caching")

# Long enough that a miss is easy to tell from a hit.
LEAF_SECONDS = 3


@env.task(cache="auto")
def summarize(groups: dict[str, list[int]], scale: float) -> dict[str, dict[str, float]]:
    print(f"summarize: computing {len(groups)} groups, scale {scale}")
    time.sleep(LEAF_SECONDS)
    stats = {}
    for name in sorted(groups):
        values = groups[name]
        stats[name] = {
            "count": float(len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": sum(values) / len(values) * scale,
        }
    return stats


# An explicit cache version: bump it to drop what is cached, whatever the code.
@env.task(cache=flyte.Cache(behavior="override", version_override="v1"))
def label(words: list[str], flags: dict[str, bool], limit: Optional[int]) -> list[str]:
    print(f"label: labelling {len(words)} words, limit {limit}")
    time.sleep(LEAF_SECONDS)
    labels = []
    for word in words:
        text = word.upper() if flags["upper"] else word
        if flags["reverse"]:
            text = text[::-1]
        labels.append(f"{text} ({len(word)})")
    return labels if limit is None else labels[:limit]


# `note` is not part of the cache key: calls that differ only in it share a result.
@env.task(cache=flyte.Cache(behavior="auto", ignored_inputs=("note",)))
def matrix_total(matrix: list[list[int]], note: str) -> int:
    print(f"matrix_total: summing {len(matrix)} rows ({note})")
    time.sleep(LEAF_SECONDS)
    return sum(sum(row) for row in matrix)


# The two drivers have the same body, so they make the same calls with the same inputs.


@env.task
def regular_driver(
    groups: dict[str, list[int]],
    scale: float,
    words: list[str],
    flags: dict[str, bool],
    matrix: list[list[int]],
) -> dict[str, dict[str, float]]:
    stats = summarize(groups=groups, scale=scale)
    stats_again = summarize(groups=groups, scale=scale)
    labels = label(words=words, flags=flags, limit=None)
    first_two = label(words=words, flags=flags, limit=2)
    total = matrix_total(matrix=matrix, note="first call")
    total_renoted = matrix_total(matrix=matrix, note="same matrix, another note")
    stats["checks"] = {
        "repeat_matches": 1.0 if stats_again == stats else 0.0,
        "labels": float(len(labels)),
        "first_two": float(len(first_two)),
        "matrix_total": float(total),
        "renoted_matches": 1.0 if total_renoted == total else 0.0,
    }
    return stats


@flyte.orchestrator.task
def orchestrator_driver(
    groups: dict[str, list[int]],
    scale: float,
    words: list[str],
    flags: dict[str, bool],
    matrix: list[list[int]],
) -> dict[str, dict[str, float]]:
    stats = summarize(groups=groups, scale=scale)
    stats_again = summarize(groups=groups, scale=scale)
    labels = label(words=words, flags=flags, limit=None)
    first_two = label(words=words, flags=flags, limit=2)
    total = matrix_total(matrix=matrix, note="first call")
    total_renoted = matrix_total(matrix=matrix, note="same matrix, another note")
    stats["checks"] = {
        "repeat_matches": 1.0 if stats_again == stats else 0.0,
        "labels": float(len(labels)),
        "first_two": float(len(first_two)),
        "matrix_total": float(total),
        "renoted_matches": 1.0 if total_renoted == total else 0.0,
    }
    return stats


def inputs(seed: int) -> dict:
    return {
        "groups": {
            "small": [3, 1, 4, 1, 5],
            "negative": [-7, -2, -9],
            # The largest and smallest 64-bit ints, which is as far as a Flyte int goes.
            "extremes": [2**63 - 1, -(2**63), 0],
            "seed": [seed],
        },
        "scale": 1.5,
        "words": ["naïve", "Ünïcödé", "日本語", "emoji 🚀", "", f"seed-{seed}"],
        "flags": {"upper": True, "reverse": False},
        "matrix": [[1, -2, 3], [], [10**12, -(10**12)], [seed]],
    }


def run_and_report(driver, values: dict, local: bool) -> dict:
    start = time.monotonic()
    run = flyte.run(driver, **values)
    print(f"{driver.name}: {run.url}")
    run.wait(quiet=True)
    print(f"  finished in {time.monotonic() - start:.1f}s")
    if not local:
        from flyteidl2.core import catalog_pb2

        for action in flyte.remote.Action.listall(for_run_name=run.name, parent_name="a0"):
            status = catalog_pb2.CatalogCacheStatus.Name(action.pb2.status.cache_status)
            print(f"  {action.name}  {action.task_name}  {status}")
    return run.outputs()[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=int(time.time()), help="the same seed reuses an earlier cache")
    parser.add_argument("--orchestrator-first", action="store_true", help="run the orchestrator driver first")
    parser.add_argument("--local", action="store_true", help="run in this process instead of on the cluster")
    args = parser.parse_args()

    if args.local:
        flyte.init()
    else:
        flyte.init_from_config()
    drivers = [regular_driver, orchestrator_driver]
    if args.orchestrator_first:
        drivers.reverse()
    values = inputs(args.seed)
    outputs = [run_and_report(driver, values, args.local) for driver in drivers]
    print("same outputs:", outputs[0] == outputs[1])
    print(outputs[1])
