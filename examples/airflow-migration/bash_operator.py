from pathlib import Path

from flyteplugins.airflow.task import AirflowContainerTask  # triggers DAG + operator patches  # type: ignore
from airflow import DAG
from airflow.operators.bash import BashOperator
import flyte

env = flyte.TaskEnvironment(
    name="hello_airflow",
    image=flyte.Image.from_debian_base().with_pip_packages("apache-airflow<3.0.0", "jsonpickle").with_local_v2()
)
# Standard Airflow DAG definition — no Flyte-specific changes needed inside the block.
# Pass flyte_env so the generated workflow task uses the right container image.
with DAG(
    dag_id='simple_bash_operator_example',
    flyte_env=env,
) as dag:
    t1 = BashOperator(
        task_id='say_hello',
        bash_command='echo "Hello Airflow1!"',
    )
    t2 = BashOperator(
        task_id='say_goodbye',
        bash_command='echo "Goodbye Airflow2!"',
    )
    # t1 >> t2  # t2 runs after t1


if __name__ == '__main__':
    flyte.init_from_config(root_dir=Path("/Users/kevin/git/flyte-sdk"))
    # dag.run() is a convenience wrapper — equivalent to:
    run = flyte.with_runcontext(mode="remote", log_level="10").run(dag)
    # run = dag.run(mode="local", log_level="10")
    print(run.url)
