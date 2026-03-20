# Pandera + Flyte examples

The **`flyteplugins-pandera`** plugin currently registers **`pandera.typing.pandas.DataFrame`** only. Use:

| File | Status |
|------|--------|
| `pandas_schema.py` | Supported — `pandera.typing.pandas.DataFrame` |

Other `*_schema.py` files (polars, ibis, Spark, modin, dask) are **not** wired through the plugin until those backends are added to `flyteplugins-pandera`.

Run from repo root (with dependencies installed). Each script accepts `--mode {local,remote}` (default: `remote`):

```bash
python examples/plugins/pandera/pandas_schema.py
python examples/plugins/pandera/pandas_schema.py --mode local
```

See [Pandera docs](https://pandera.readthedocs.io/en/latest/) for schema syntax.
