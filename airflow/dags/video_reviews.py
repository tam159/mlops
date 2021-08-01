"""Airflow DAG for Amazon video reviews."""

import boto3
import sagemaker
import video_config as cfg
from airflow.models import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.amazon.aws.operators.sagemaker_training import (
    SageMakerTrainingOperator,
)
from airflow.providers.amazon.aws.operators.sagemaker_transform import (
    SageMakerTransformOperator,
)
from airflow.providers.amazon.aws.operators.sagemaker_tuning import (
    SageMakerTuningOperator,
)
from airflow.utils.dates import days_ago
from processing import video_reviews_prepare, video_reviews_preprocess
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.airflow import (
    training_config,
    transform_config_from_estimator,
    tuning_config,
)

# =============================================================================
# functions
# =============================================================================


def is_hpo_enabled():
    """Check if hyper-parameter optimization is enabled in the config."""
    is_hpo = False
    if "job_level" in config and "run_hyperparameter_opt" in config["job_level"]:
        run_hpo_config = config["job_level"]["run_hyperparameter_opt"]
        if run_hpo_config.lower() == "yes":
            is_hpo = True
    return is_hpo


def get_sagemaker_role_arn(role_name: str, region_name: str) -> str:
    """
    Get sagemaker role arn.

    :param role_name: role name
    :param region_name: region name
    :return: role arn
    """
    iam = boto3.client("iam", region_name=region_name)
    response = iam.get_role(RoleName=role_name)
    return response["Role"]["Arn"]


# =============================================================================
# setting up training, tuning and transform configuration
# =============================================================================


# read config file
config = cfg.config

# set configuration for tasks
region = config["job_level"]["region_name"]
role = get_sagemaker_role_arn(config["train_model"]["sagemaker_role"], region)

sagemaker_session = sagemaker.Session()
container = image_uris.retrieve("factorization-machines", region)
hpo_enabled = is_hpo_enabled()

# create estimator
fm_estimator = Estimator(
    image_uri=container,
    role=role,
    sagemaker_session=sagemaker_session,
    **config["train_model"]["estimator_config"]
)

# train_config specifies SageMaker training configuration
train_config = training_config(
    estimator=fm_estimator, inputs=config["train_model"]["inputs"]
)

# create tuner
fm_tuner = HyperparameterTuner(
    estimator=fm_estimator, **config["tune_model"]["tuner_config"]
)

# create tuning config
tuner_config = tuning_config(tuner=fm_tuner, inputs=config["tune_model"]["inputs"])

# create transform config
transform_config = transform_config_from_estimator(
    estimator=fm_estimator,
    task_id="model_tuning" if hpo_enabled else "model_training",
    task_type="tuning" if hpo_enabled else "training",
    **config["batch_transform"]["transform_config"]
)

# =============================================================================
# define airflow DAG and tasks
# =============================================================================

# define airflow DAG

default_args = {
    "start_date": days_ago(2),
    "provide_context": True,
}

with DAG(
    dag_id="video-reviews",
    default_args=default_args,
    schedule_interval="@once",
    concurrency=1,
    max_active_runs=1,
) as dag:
    # preprocess the data
    preprocess_task = PythonOperator(
        task_id="preprocessing",
        python_callable=video_reviews_preprocess.preprocess,
        op_kwargs=config["preprocess_data"],
    )

    # prepare the data for training
    prepare_task = PythonOperator(
        task_id="preparing",
        python_callable=video_reviews_prepare.prepare,
        op_kwargs=config["prepare_data"],
    )

    branching = BranchPythonOperator(
        task_id="branching",
        python_callable=lambda: "model_tuning" if hpo_enabled else "model_training",
    )

    # launch sagemaker training job and wait until it completes
    train_model_task = SageMakerTrainingOperator(
        task_id="model_training",
        config=train_config,
        wait_for_completion=True,
        check_interval=30,
    )

    # launch sagemaker hyperparameter job and wait until it completes
    tune_model_task = SageMakerTuningOperator(
        task_id="model_tuning",
        config=tuner_config,
        wait_for_completion=True,
        check_interval=30,
    )

    # launch sagemaker batch transform job and wait until it completes
    batch_transform_task = SageMakerTransformOperator(
        task_id="predicting",
        config=transform_config,
        wait_for_completion=True,
        check_interval=30,
        trigger_rule="one_success",
    )

    # set the dependencies between tasks
    preprocess_task >> prepare_task >> branching
    branching >> [train_model_task, tune_model_task] >> batch_transform_task
