# Flyte Airflow Plugin

Run existing Airflow DAGs on Flyte with minimal code changes. The plugin
monkey-patches `airflow.DAG` and `BaseOperator` so that standard Airflow
definitions are transparently converted into Flyte tasks.

## Features

- Write a normal `with DAG(...) as dag:` block — the plugin intercepts
  operator construction and wires everything into a Flyte workflow.
- Supports `BashOperator` (`AirflowShellTask`) and `PythonOperator`
  (`AirflowPythonFunctionTask`).
- Dependency arrows (`>>`, `<<`) are preserved as execution order.
- Runs locally or remotely — no Airflow cluster required.

## Installation

```bash
pip install flyteplugins-airflow
```

## Quick start

```python
import flyteplugins.airflow.task  # triggers DAG + operator monkey-patches
from airflow import DAG
from airflow.operators.bash import BashOperator
import flyte

with DAG(dag_id="my_dag") as dag:
    t1 = BashOperator(task_id="step1", bash_command="echo step1")
    t2 = BashOperator(task_id="step2", bash_command="echo step2")
    t1 >> t2

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote").run(dag)
    print(run.url)
```

See `examples/airflow-migration/` for a full example including `PythonOperator`.
