"""Airflow custom hook test."""

from airflow.hooks.base import BaseHook


class MyHook(BaseHook):
    """Testing hook."""

    def __init__(self, my_field: str) -> None:
        """Init method."""
        self.my_field = my_field

    def my_method(self):
        """Test method."""
        print(self.my_field)
