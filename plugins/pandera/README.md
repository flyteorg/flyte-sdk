# Flyte Pandera Plugin

`flyteplugins-pandera` adds support for pandera dataframe typing hints in Flyte v2.

Supported pandera typing types (install the matching [Pandera extra](https://pandera.readthedocs.io/en/latest/#extras) for your dataframe library):

| Pandera typing | Install |
| --- | --- |
| `pandera.typing.pandas.DataFrame` | `pip install flyteplugins-pandera 'pandera[pandas]'` |
| `pandera.typing.polars.DataFrame` | `pip install flyteplugins-pandera 'pandera[polars]'` |
| `pandera.typing.polars.LazyFrame` | `pip install flyteplugins-pandera 'pandera[polars]'` |
| `pandera.typing.ibis.Table` | `pip install flyteplugins-pandera 'pandera[ibis]'` |
| `pandera.typing.pyspark_sql.DataFrame` | `pip install flyteplugins-pandera 'pandera[pyspark]'` |
| `pandera.typing.pyspark.DataFrame` | `pip install flyteplugins-pandera 'pandera[pyspark]'` |
| `pandera.typing.modin.DataFrame` | `pip install flyteplugins-pandera 'pandera[modin]'` |
| `pandera.typing.dask.DataFrame` | `pip install flyteplugins-pandera 'pandera[dask]'` |

At runtime, the plugin:

1. delegates dataframe IO to Flyte's `DataFrameTransformerEngine`,
2. validates data with pandera schemas, and
3. writes a validation report to `flyte.report`.
