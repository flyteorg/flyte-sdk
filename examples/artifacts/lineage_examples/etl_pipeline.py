"""ETL use case: two independent sources converge, then load to a warehouse table.

A diamond-shaped lineage: `extract_orders` and `extract_customers` are two
unrelated producers with no shared ancestor; `join_orders_customers` is a merge
point that consumes one artifact from each side of the diamond. `load_warehouse`
adds one more hop, and `etl_monitor_app` demonstrates the app-consumption
fallback again, this time watching the final loaded table.

Try it:

    python examples/artifacts/lineage_examples/etl_pipeline.py
"""

import csv
import io
import tempfile
import time

import fastapi
from lineage import LABEL_UPSTREAM_ARTIFACT_NAME, LABEL_UPSTREAM_ARTIFACT_VERSION

import flyte
import flyte.artifacts as artifacts
from flyte.app.extras import FastAPIAppEnvironment
from flyte.io import File

# Shared by the task env and the monitor app below: the module imports `fastapi` at top
# level (for `etl_app`), so any task defined here needs it too, even ones that never
# touch FastAPI themselves -- task loading imports the whole module.
image = flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn")
env = flyte.TaskEnvironment(name="etl_pipeline", image=image)


def _wait_for_artifact(name: str, *, retries: int = 15, delay_s: float = 2.0):
    """`Artifact.get` right after a run completes can race artifact indexing; retry briefly."""
    from flyte.remote import Artifact

    for attempt in range(retries):
        try:
            return Artifact.get(name)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay_s)


def _run_and_wait(run):
    """`.wait()` alone doesn't raise on failure -- surface it immediately instead of
    letting a later `_wait_for_artifact` call fail with a confusing "not found"."""
    run.wait()
    if run.phase != "succeeded":
        raise RuntimeError(f"{run.url} finished in phase {run.phase!r}")
    return run


def _write_csv(rows: list[dict], fieldnames: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(buf.getvalue())
        return f.name


async def _read_csv(file: File) -> list[dict]:
    async with file.open("rb") as fh:
        text = bytes(await fh.read()).decode()
    return list(csv.DictReader(io.StringIO(text)))


@env.task(produces_artifacts=True)
async def extract_orders() -> File:
    rows = [
        {"order_id": "1", "customer_id": "c1", "amount": "19.99"},
        {"order_id": "2", "customer_id": "c2", "amount": "42.00"},
        {"order_id": "3", "customer_id": "c1", "amount": "8.50"},
    ]
    path = _write_csv(rows, ["order_id", "customer_id", "amount"])
    file = await File.from_local(path)
    return artifacts.new(file, artifacts.Metadata(name="raw-orders", description="Raw orders extract"))


@env.task(produces_artifacts=True)
async def extract_customers() -> File:
    rows = [
        {"customer_id": "c1", "name": "Ada", "region": "us"},
        {"customer_id": "c2", "name": "Grace", "region": "eu"},
    ]
    path = _write_csv(rows, ["customer_id", "name", "region"])
    file = await File.from_local(path)
    return artifacts.new(file, artifacts.Metadata(name="raw-customers", description="Raw customers extract"))


@env.task(produces_artifacts=True)
async def transform_orders(orders: File) -> File:
    rows = await _read_csv(orders)
    for row in rows:
        row["amount"] = f"{float(row['amount']):.2f}"
    path = _write_csv(rows, ["order_id", "customer_id", "amount"])
    file = await File.from_local(path)
    return artifacts.new(file, artifacts.Metadata(name="clean-orders", description="Normalized orders"))


# Merge point: one artifact from each side of the diamond. Both edges are found via
# the automatic bound-input scan.
@env.task(produces_artifacts=True)
async def join_orders_customers(orders: File, customers: File) -> File:
    order_rows = await _read_csv(orders)
    customer_by_id = {c["customer_id"]: c for c in await _read_csv(customers)}
    joined = [
        {
            **row,
            "name": customer_by_id[row["customer_id"]]["name"],
            "region": customer_by_id[row["customer_id"]]["region"],
        }
        for row in order_rows
    ]
    path = _write_csv(joined, ["order_id", "customer_id", "amount", "name", "region"])
    file = await File.from_local(path)
    return artifacts.new(file, artifacts.Metadata(name="orders-enriched", description="Orders joined with customers"))


@env.task(produces_artifacts=True)
async def load_warehouse(enriched: File) -> File:
    rows = await _read_csv(enriched)
    path = _write_csv(rows, ["order_id", "customer_id", "amount", "name", "region"])
    file = await File.from_local(path)
    metadata = artifacts.Metadata(
        name="warehouse-orders-table", description="Loaded orders table", data={"rows": str(len(rows))}
    )
    return artifacts.new(file, metadata)


etl_app = fastapi.FastAPI(title="ETL Monitor")


@etl_app.get("/status")
async def status() -> dict:
    from flyte.remote import Artifact

    table = await Artifact.get.aio(name="warehouse-orders-table")
    return {"table": table.name, "version": table.version, "rows": table.pb2.spec.info.user_metadata.get("rows")}


# Fetches the table artifact's metadata by name at request time -- the labels (set on
# this module-level object below, once the table's identity is known) make it
# discoverable as a consumer, same fallback as the ML example's serving app. Setting
# `.labels` in place (not `clone_with()`) keeps the object the resolver looks up by name.
etl_monitor_app = FastAPIAppEnvironment(name="etl-monitor-app", app=etl_app, image=image)


def run_pipeline() -> None:
    """Extract both sources, transform, join, load, and serve. Assumes init_from_config() ran."""
    orders_run = flyte.run(extract_orders)
    print(orders_run.url)
    customers_run = flyte.run(extract_customers)
    print(customers_run.url)
    _run_and_wait(orders_run)
    _run_and_wait(customers_run)
    orders = _wait_for_artifact("raw-orders")
    customers = _wait_for_artifact("raw-customers")

    transform_run = _run_and_wait(flyte.run(transform_orders, orders=orders))
    print(transform_run.url)
    clean_orders = _wait_for_artifact("clean-orders")

    join_run = _run_and_wait(flyte.run(join_orders_customers, orders=clean_orders, customers=customers))
    print(join_run.url)
    enriched = _wait_for_artifact("orders-enriched")

    load_run = _run_and_wait(flyte.run(load_warehouse, enriched=enriched))
    print(load_run.url)
    table = _wait_for_artifact("warehouse-orders-table")

    etl_monitor_app.labels = {LABEL_UPSTREAM_ARTIFACT_NAME: table.name, LABEL_UPSTREAM_ARTIFACT_VERSION: table.version}
    flyte.serve(etl_monitor_app)


if __name__ == "__main__":
    flyte.init_from_config()
    run_pipeline()
