"""Airflow custom operator test."""

from airflow.models.baseoperator import BaseOperator


class HelloOperator(BaseOperator):
    """Testing hello operator."""

    def __init__(self, name: str, **kwargs) -> None:
        """Init method."""
        super().__init__(**kwargs)
        self.name = name

    def execute(self, context):
        """Execute method."""
        message = "Hello {}".format(self.name)
        print(message)
        return message
