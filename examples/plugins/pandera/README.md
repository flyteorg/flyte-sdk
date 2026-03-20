# Pandera + Flyte examples

Each `*_schema.py` demonstrates `flyteplugins-pandera` with `pandera.typing` for one dataframe backend.

| File | Backend | Notes |
|------|---------|--------|
| `pandas_schema.py` | `pandera.typing.pandas.DataFrame` | Uses built-in pandas parquet handlers. |
| `polars_schema.py` | `pandera.typing.polars.DataFrame` / `LazyFrame` | Requires `flyteplugins-polars`. |
| `ibis_schema.py` | `pandera.typing.ibis.Table` | Requires `pandera[ibis]`; Flyte may need a custom `ibis.Table` encoder for full remote I/O. |
| `pyspark_sql_schema.py` | `pandera.typing.pyspark_sql.DataFrame` | Requires `flyteplugins-spark` and a Spark-enabled task environment. |
| `pyspark_schema.py` | `pandera.typing.pyspark.DataFrame` (pandas-on-Spark) | Same Spark setup as above. |
| `modin_schema.py` | `pandera.typing.modin.DataFrame` | Requires `pandera` + Modin; add a Modin dataframe encoder for remote round-trips if needed. |
| `dask_schema.py` | `pandera.typing.dask.DataFrame` | Requires `pandera` + Dask; add a Dask dataframe encoder for remote round-trips if needed. |

Run from repo root (with dependencies installed). Each script accepts `--mode {local,remote}` (default: `remote`):

```bash
python examples/plugins/pandera/pandas_schema.py
python examples/plugins/pandera/pandas_schema.py --mode local
```

See [Pandera docs](https://pandera.readthedocs.io/en/latest/) for schema syntax per backend.
