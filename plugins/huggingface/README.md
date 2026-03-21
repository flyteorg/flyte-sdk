# Hugging Face Datasets Plugin

This plugin provides native support for **Hugging Face Datasets** in Flyte, enabling seamless serialization and deserialization of `datasets.Dataset` objects through Parquet format.

The plugin supports:
- `datasets.Dataset` - Hugging Face's dataset type for structured data
- Automatic column filtering based on type annotations
- Efficient Parquet-based storage for large datasets

To install the plugin, run the following command:

```bash
pip install --pre flyteplugins-huggingface
```
