# from flyteplugins.airflow.task import AirflowContainerTask
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
)


@env.task
async def fn(name: str) -> None:
    print("starting to run airflow task")
    # ExecutorSafeguard stores a sentinel in a threading.local() dict. That dict
    # is initialised on the main thread at import time, but Flyte runs tasks in a
    # background async thread where the thread-local has no 'callers' key yet.
    if not hasattr(ExecutorSafeguard._sentinel, "callers"):
        ExecutorSafeguard._sentinel.callers = {}
    BashOperator(
        task_id='airflow',
        bash_command=f'echo "Hello {name}!"',
    ).execute(context=airflow_context.Context())
    print("finished running airflow task")


if __name__ == '__main__':
    flyte.init_from_config()
    run = flyte.with_runcontext(mode="local", log_level="10").run(fn, name="Airflow")
    print(run.url)
