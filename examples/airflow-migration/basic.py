from pathlib import Path

import flyteplugins.airflow.task  # noqa: F401 — triggers DAG + operator monkey-patches
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import flyte


def hello_python():
    print("Hello from PythonOperator!")


# Standard Airflow DAG definition — no Flyte-specific changes needed inside the block.
# Pass flyte_env so the generated workflow task uses the right container image.
with DAG(
    dag_id="simple_airflow_workflow",
) as dag:
    t1 = BashOperator(
        task_id="say_hello",
        bash_command='echo "Hello Airflow!"',
    )
    t2 = BashOperator(
        task_id="say_goodbye",
        bash_command='echo "Goodbye Airflow!"',
    )
    t3 = PythonOperator(
        task_id="hello_python",
        python_callable=hello_python,
    )
    t1 >> t2  # t2 runs after t1


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent.parent.parent)
    run = flyte.with_runcontext(mode="remote", log_level="10").run(dag)
    print(run.url)
