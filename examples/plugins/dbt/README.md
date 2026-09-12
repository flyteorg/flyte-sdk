# dbt plugin example

This example runs the dbt jaffle shop project with `flyteplugins-dbt`.

The included profile uses DuckDB so the example can run without external database credentials. Because DuckDB writes to a local file inside the task container, the example uses one `dbt build` invocation to seed data, build models, and run tests in the same task.

If your dbt project uses a shared warehouse, you can split commands such as `dbt seed`, `dbt run`, and `dbt test` into separate `DbtTask` invocations.

```bash
python dbt_build_example.py
```
