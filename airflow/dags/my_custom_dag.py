"""Airflow DAG for a custom plugin test."""

from datetime import datetime, timedelta

from operators.hello_operator import HelloOperator
from operators.my_airflow_operator import MyOperator
from sensors.my_airflow_sensor import MySensor

from airflow import DAG

default_args = {
    "depends_on_past": False,
    "start_date": datetime(2021, 6, 2),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "custom_dag",
    max_active_runs=3,
    schedule_interval="@once",
    default_args=default_args,
) as dag:
    sens = MySensor(task_id="sensor_task")

    op = MyOperator(
        task_id="custom_operator_hook_task", my_field="Hello custom hook operator"
    )

    hello_task = HelloOperator(task_id="custom_operator_task", name="custom operator")

    sens >> op >> hello_task
