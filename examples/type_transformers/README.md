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
it would not be loaded without intervention because there's no `import my_pandas_transformer` statement anywhere. This is
not the case when it comes to task plugins - this is because in order to use a task plugin, you have to declare it in
the `TaskEnvironment`, for example Spark

```python
from flyteplugins.spark.task import Spark

spark_env = flyte.TaskEnvironment(
    name="spark_env",
    resources=flyte.Resources(cpu=(1, 2), memory=("3000Mi", "5000Mi")),
    plugin_config=Spark(spark_conf=...),
)
```
Any corresponding type plugins should've been declared and loaded in the spark plugin package.

To remediate this issue for pure type plugins, Flyte provides an intervention mechanism in the form of a Python importlib entry_point.
Register your type plugin with group name `flyte.plugins.types` and Flyte will by default, automatically load it on startup.

```yaml
[project.entry-points."flyte.plugins.types"]
my_transformer = "my_transformer.transformer:register_positive_int_transformer"
```

If the entry_point object is a callable function, it will be called at load time with no arguments. To avoid loading and initializing
type plugins, call `flyte.init` with `load_plugin_type_transformers=False`.

This folder contains an example type plugin, as well as a main.py that leverages it. To run the example, run the
`__main__` block in the `main.py` file - running from the CLI isn't possible yet because there's currently no way to
express how custom types should be parsed from the command line string.
