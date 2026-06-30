# Polars Plugin

This plugin provides native support for **Polars DataFrames and LazyFrames** in Flyte, enabling efficient data processing with Polars' high-performance DataFrame library.

The plugin supports:
- `polars.DataFrame` - Eager evaluation DataFrames
- `polars.LazyFrame` - Lazy evaluation DataFrames for optimized query execution

Both types can be serialized to and deserialized from Parquet format, making them ideal for large-scale data processing workflows.

To install the plugin, run the following command:

```bash
pip install --pre flyteplugins-polars
```
