"""Airflow custom plugin test."""

from airflow.plugins_manager import AirflowPlugin
from hooks.my_airflow_hook import MyHook
from operators.my_airflow_operator import MyOperator
from sensors.my_airflow_sensor import MySensor


class PluginName(AirflowPlugin):
    """Testing plugin."""

    name = "my_airflow_plugin"

    hooks = [MyHook]
    operators = [MyOperator]
    sensors = [MySensor]
