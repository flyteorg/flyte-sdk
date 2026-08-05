# Flyte Pandera Plugin

`flyteplugins-pandera` adds support for **`pandera.typing.pandas.DataFrame`**, **`pandera.typing.polars.DataFrame` / `LazyFrame`**, and **`pandera.typing.pyspark_sql.DataFrame`** in Flyte v2.

Install:

```bash
pip install flyteplugins-pandera 'pandera[pandas]'            # pandas only
pip install flyteplugins-pandera 'pandera[polars]' flyteplugins-polars   # Polars + structured dataset I/O
pip install flyteplugins-pandera 'pandera[pyspark]' flyteplugins-spark     # PySpark SQL + parquet I/O
```

For PySpark, structured dataset serialization uses Flyte’s `DataFrameTransformerEngine` parquet handlers from **`flyteplugins-spark`** (register Spark encoders/decoders alongside this plugin).

At runtime, the plugin:

1. delegates dataframe IO to Flyte's `DataFrameTransformerEngine`,
2. validates data with pandera schemas, and
3. writes a validation report to `flyte.report`.

Validation **always** runs on every encode/decode. Report tabs are **suppressed** automatically when Flyte is only moving literals across a nested-task boundary (parent task encoding child inputs, or materializing a child’s outputs inside the parent). The SDK sets `TaskContext.in_driver_literal_conversion` on the active task (check with `flyte.ctx()` when non-`None`, then `.in_driver_literal_conversion`) so you see one report per dataframe on the task that actually produced or consumed it as task body I/O, not extra tabs on the orchestrating “driver” task.

### Troubleshooting

If logs show **“Unsupported Type pandera.typing… Flyte will default to use PickleFile”**, the pandera transformer was not registered:

- Install the plugin in **every** environment (local runner and task image): `pip install flyteplugins-pandera`.
- Flyte loads `flyte.plugins.types` during `flyte.initialize()` and on first `TypeEngine` use; confirm the distribution is installed (`import importlib.metadata as m; print(list(m.entry_points(group="flyte.plugins.types")))`).
- **Import order:** import your `pandera.typing.*` modules **before** plugin registration runs in files that run early (tests, `__init__.py`). Loading the plugin before pandera can leave two different `pandera.typing.pandas.DataFrame` (or polars container) class objects in the process; `TypeEngine` would only know about one of them, so annotations on the other fall through to pickle / the generic handler.
