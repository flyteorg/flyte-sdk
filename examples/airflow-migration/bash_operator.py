from airflow import DAG
from airflow.operators.bash import BashOperator
from pendulum import datetime

with DAG(
    dag_id='simple_bash_operator_example',
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    # Define the BashOperator task
    hello_task = BashOperator(
        task_id='say_hello',
        bash_command='echo "Hello Airflow!"',
    )
