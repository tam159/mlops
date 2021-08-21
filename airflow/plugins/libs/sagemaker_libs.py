"""Sagemaker libraries."""

from typing import Dict, List

from sagemaker.estimator import EstimatorBase
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor


def run_process_job(
    processor: ScriptProcessor,
    code: str,
    inputs: List[ProcessingInput],
    outputs: List[ProcessingOutput],
    arguments: List[str],
) -> None:
    """
    Run a SageMaker processing job.

    :param processor: processor
    :param code: code path to run the job
    :param inputs: job processing inputs
    :param outputs: job processing outputs
    :param arguments: job arguments
    :return: None
    """
    processor.run(
        code=code,
        inputs=inputs,
        outputs=outputs,
        arguments=arguments,
    )


def run_train_job(estimator: EstimatorBase, data_channels: Dict) -> None:
    """
    Run a SageMaker training job.

    :param estimator: estimator
    :param data_channels: data channels including train, validation, test
    """
    estimator.fit(data_channels)
