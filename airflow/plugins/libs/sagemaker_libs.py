"""Sagemaker libraries."""

from typing import List

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor


def run_process_job(
    processor: ScriptProcessor,
    code: str,
    inputs: List[ProcessingInput],
    outputs: List[ProcessingOutput],
    arguments: List[str],
) -> None:
    """
    Run processing job.

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
