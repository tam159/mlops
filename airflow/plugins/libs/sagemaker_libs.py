"""Sagemaker libraries."""

from typing import List

import libs.project_config as cfg
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor


def run_process_job(
    processor: ScriptProcessor,
    code: str,
    input_path: str,
    output_path: str,
    arguments: List[str],
) -> None:
    """
    Run processing job.

    :param processor: processor
    :param code: code path to run the job
    :param input_path: input data path
    :param output_path: output data path
    :param arguments: job arguments
    :return: None
    """
    processor.run(
        code=code,
        inputs=[ProcessingInput(source=input_path, destination=cfg.container_input)],
        outputs=[
            ProcessingOutput(
                source=cfg.container_train_output, destination=f"{output_path}/train"
            ),
            ProcessingOutput(
                source=cfg.container_test_output, destination=f"{output_path}/test"
            ),
        ],
        arguments=arguments,
    )
