"""Example 5 — Parquet vs Lance, through flyte.io.DataFrame.

``flyte.io.DataFrame`` stores a ``pyarrow.Table`` as Parquet by default; this
plugin adds Lance. Both are columnar and Arrow-native, so reading all the rows
costs about the same. The difference is random access:

- The built-in Parquet DataFrame decodes eagerly — to get a few scattered rows you
  ``open(pa.Table).all()`` the whole table into memory, then index it.
- The Lance DataFrame hands you a ``lance.LanceDataset`` and you ``take([...])`` —
  it reads only the rows you asked for.

So for a shuffled training batch, Parquet reads the entire dataset while Lance
reads ~the batch. This example builds the same data as both formats (through the
plugin), then benchmarks fetching a random batch from each and renders a Flyte
report. The headline metric is *data read*: it's cache-independent and it's what
turns into a latency gap on real object storage.

Run:  python examples/05_parquet_vs_lance.py
"""

import random
import time
from typing import Annotated

import flyte
import flyte.report
import lance
import pyarrow as pa
from flyte.io import DataFrame

env = flyte.TaskEnvironment(
    name="lance-ex-compare",
    image=flyte.Image.from_debian_base(name="lance-examples").with_local_v2_plugins("flyteplugins-lance"),
)


def _build_table(n_rows: int, payload_bytes: int) -> pa.Table:
    """id + a float feature + a binary payload (stand-in for an embedding / image),
    so a random-row fetch moves real bytes rather than just integers."""
    rng = random.Random(0)
    return pa.table(
        {
            "id": pa.array(range(n_rows), type=pa.int64()),
            "x": pa.array([rng.random() for _ in range(n_rows)], type=pa.float64()),
            "payload": pa.array([rng.randbytes(payload_bytes) for _ in range(n_rows)], type=pa.large_binary()),
        }
    )


@env.task
async def make_parquet(n_rows: int = 100_000, payload_bytes: int = 512) -> DataFrame:
    """Return a DataFrame with no format annotation -> stored as Parquet (the default)."""
    return DataFrame.from_df(_build_table(n_rows, payload_bytes))


@env.task
async def make_lance(n_rows: int = 100_000, payload_bytes: int = 512) -> Annotated[DataFrame, "lance"]:
    """Return the same data annotated as "lance" -> stored by this plugin."""
    return DataFrame.from_df(_build_table(n_rows, payload_bytes))


def _best_time(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


async def _best_time_async(coro_fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        await coro_fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _bar(label: str, value: float, max_value: float, unit: str, color: str) -> str:
    pct = 100 * value / max_value if max_value else 0
    return (
        '<div style="margin:8px 0;font-family:system-ui,sans-serif">'
        f'<div style="font-size:13px;margin-bottom:3px">{label} &mdash; <b>{value:,.0f}{unit}</b></div>'
        '<div style="background:#e9ecef;border-radius:5px;overflow:hidden">'
        f'<div style="width:{pct:.1f}%;background:{color};height:18px"></div></div></div>'
    )


def _section(title: str, subtitle: str, parquet_value: float, lance_value: float, unit: str) -> str:
    max_value = max(parquet_value, lance_value) or 1
    return (
        f'<h3 style="font-family:system-ui,sans-serif;margin:20px 0 2px">{title}</h3>'
        f'<p style="font-family:system-ui,sans-serif;color:#666;font-size:13px;margin:0 0 6px">{subtitle}</p>'
        + _bar("Parquet", parquet_value, max_value, unit, "#adb5bd")
        + _bar("Lance", lance_value, max_value, unit, "#06d6a0")
    )


@env.task(report=True)
async def compare(parquet_df: DataFrame, lance_ds: lance.LanceDataset, n_random_rows: int = 1_000) -> dict:
    n = lance_ds.count_rows()
    indices = random.Random(7).sample(range(n), k=min(n_random_rows, n))
    k = len(indices)
    cols = ["id", "x", "payload"]

    # Random access: fetch k scattered rows (a shuffled batch)
    # Lance reads only those rows straight from the handle.
    lance_rows = lance_ds.take(indices, columns=cols)
    lance_read_mb = lance_rows.nbytes / 1e6
    lance_random_s = _best_time(lambda: lance_ds.take(indices, columns=cols))

    # The Parquet DataFrame decodes eagerly: materialize the whole table, then index.
    full_table = await parquet_df.open(pa.Table).all()
    parquet_read_mb = full_table.nbytes / 1e6  # all n rows, just to return k

    async def _parquet_random():
        table = await parquet_df.open(pa.Table).all()
        table.take(indices)

    parquet_random_s = await _best_time_async(_parquet_random)

    data_ratio = parquet_read_mb / lance_read_mb if lance_read_mb else float("inf")
    time_ratio = parquet_random_s / lance_random_s if lance_random_s else float("inf")
    rnd_pq_tps, rnd_lance_tps = k / parquet_random_s, k / lance_random_s

    html = f"""
    <div style="font-family:system-ui,sans-serif;max-width:720px">
    <h2>Parquet vs Lance &mdash; fetching a shuffled batch from {n:,} rows</h2>
    <p style="color:#444">Both hold the same columnar data via <code>flyte.io.DataFrame</code>
    (Parquet is the built-in default; Lance comes from this plugin). What differs is the cost
    of pulling out {k:,} scattered rows &mdash; a shuffled training batch.</p>

    <div style="margin:14px 0;padding:12px 14px;background:#e6fcf5;border-radius:8px;font-size:14px;color:#087f5b">
      To return <b>{k:,}</b> random rows, the Parquet DataFrame materialized the whole table
      (<b>{parquet_read_mb:,.0f} MB</b>); Lance read only those rows (<b>{lance_read_mb:,.1f} MB</b>)
      &mdash; <b>{data_ratio:,.0f}x less data</b>. That ratio is independent of caching, and it
      still buys a real speedup on object storage.
    </div>

    {_section(
        f"Random access &mdash; deliver {k:,} scattered rows",
        "Higher is better. Lance stays several times faster on both local disk and object storage "
        "&mdash; it reads far less data, though scattered reads there pay some per-request latency.",
        rnd_pq_tps, rnd_lance_tps, " rows/s",
    )}

    <table style="border-collapse:collapse;font-size:13px;margin-top:14px">
      <tr style="text-align:left;border-bottom:1px solid #ccc">
        <th style="padding:4px 12px">Format</th>
        <th style="padding:4px 12px">Data read for {k:,}-row fetch</th>
        <th style="padding:4px 12px">Fetch time</th>
      </tr>
      <tr><td style="padding:4px 12px">Parquet (materialize whole table)</td>
          <td style="padding:4px 12px">{parquet_read_mb:,.0f} MB</td>
          <td style="padding:4px 12px">{parquet_random_s * 1e3:,.0f} ms</td></tr>
      <tr><td style="padding:4px 12px">Lance (take just the rows)</td>
          <td style="padding:4px 12px">{lance_read_mb:,.1f} MB</td>
          <td style="padding:4px 12px">{lance_random_s * 1e3:,.1f} ms</td></tr>
    </table>

    <p style="color:#666;font-size:13px;margin-top:14px">
      Reach for Lance when you fetch scattered rows &mdash; shuffled training, point lookups &mdash;
      where reading only what you need pays off. For reading a dataset end to end both are columnar
      and fine, so Parquet stays a solid default for sequential analytics. Either way it's just a
      format on <code>flyte.io.DataFrame</code>.
    </p>
    </div>
    """
    await flyte.report.replace.aio(html, do_flush=True)

    return {
        "n_rows": n,
        "n_random_rows": k,
        "parquet_data_read_mb": round(parquet_read_mb, 1),
        "lance_data_read_mb": round(lance_read_mb, 2),
        "less_data_read_x": round(data_ratio, 1),
        "parquet_fetch_ms": round(parquet_random_s * 1e3, 1),
        "lance_fetch_ms": round(lance_random_s * 1e3, 1),
        "faster_x": round(time_ratio, 1),
    }


@env.task
async def main(n_rows: int = 100_000, payload_bytes: int = 512, n_random_rows: int = 1_000) -> dict:
    parquet_df = await make_parquet(n_rows, payload_bytes)
    lance_df = await make_lance(n_rows, payload_bytes)
    return await compare(parquet_df, lance_df, n_random_rows)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(f"Run URL: {run.url}")
