from flyteplugins.airflow.task import AirflowContainerTask  # type: ignore
from airflow import DAG
from airflow.models.baseoperator import ExecutorSafeguard
from airflow.operators.bash import BashOperator
from pendulum import datetime
import airflow.utils.context as airflow_context
import flyte

# with DAG(
#     dag_id='simple_bash_operator_example',
#     start_date=datetime(2025, 1, 1),
#     schedule=None,
#     catchup=False,
# ) as dag:
#     # Define the BashOperator task
#     hello_task = BashOperator(
#         task_id='say_hello',
#         bash_command='echo "Hello Airflow!"',
#     )


env = flyte.TaskEnvironment(
    name="hello_airflow",
    image=flyte.Image.from_debian_base().with_pip_packages("apache-airflow<3.0.0", "jsonpickle").with_local_v2()
)


@env.task
async def main(name: str) -> None:
    print("starting to run airflow task")
    BashOperator(
        task_id='airflow',
        bash_command=f'echo "Hello {name}!"',
    )
    print("finished running airflow task")


if __name__ == '__main__':
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="remote", log_level="10").run(main, name="Airflow")
    print(run.url)
