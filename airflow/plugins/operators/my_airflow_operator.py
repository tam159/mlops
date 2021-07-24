"""Airflow custom operator test."""

from airflow.operators.bash import BaseOperator
from hooks.my_airflow_hook import MyHook


class MyOperator(BaseOperator):
    """Testing hello operator."""

    def __init__(self, my_field, *args, **kwargs):
        """Init method."""
        super().__init__(*args, **kwargs)
        self.my_field = my_field

    def execute(self, context):
        """Execute method."""
        hook = MyHook(self.my_field)
        hook.my_method()
