"""Airflow custom sensor test."""

from airflow.sensors.base import BaseSensorOperator


class MySensor(BaseSensorOperator):
    """Testing hello operator."""

    def __init__(self, *args, **kwargs):
        """Init method."""
        super().__init__(*args, **kwargs)

    def poke(self, context):
        """Poke method."""
        return True
