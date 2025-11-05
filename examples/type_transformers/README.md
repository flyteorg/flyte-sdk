# Flyte Type Transformers Example

Type transformers plugins are different from a task type plugin in that often you will write Task signatures that take in
or return that type, but you would of course need to `import` that type, but there's no need to import your transformer
for the type itself.

For example, let's say you wrote a custom type transformer for `pandas.DataFrame`. You would still write your task like this:

```python
import pandas as pd
import flyte

env = flyte.TaskEnvironment(
    name="custom_types",
)

@env.task
def my_task(df: pd.DataFrame) -> int:
    ...
```

If your custom type transformer had been published to PyPI and installed in your environment under say `my_pandas_transfomer`,
it would not be loaded without intervention because there's no `import my_pandas_transformer` statement anywhere.

To fix this, Flyte provides such an intervention mechanism.

