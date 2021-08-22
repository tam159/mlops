"""Airflow DAG for retail sales prediction."""

from datetime import timedelta

from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from libs import project_config as cfg
from libs.sagemaker_libs import run_process_job, run_train_job
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost.estimator import XGBoost

folder = "retail_sales"
content_type = "csv"

source_dir = f"{cfg.sagemaker_jobs_dir}/{folder}"

# Pre process data variables
sklearn_processor = SKLearnProcessor(
    base_job_name="retail-sales-preprocess",
    framework_version="0.23-1",
    role=cfg.sagemaker_role,
    instance_type="ml.m5.large",
    instance_count=1,
)

preprocess_code = f"{source_dir}/preprocess.py"
preprocess_arguments = ["--train_ratio", "0.7", "--validation_ratio", "0.2"]

preprocess_s3_input = f"{cfg.raw_bucket}/{folder}"
preprocess_s3_output = f"{cfg.curated_bucket}/{folder}/preprocess"

preprocess_s3_train_output = f"{preprocess_s3_output}/train"
preprocess_s3_validation_output = f"{preprocess_s3_output}/validation"
preprocess_s3_test_output = f"{preprocess_s3_output}/test"

preprocess_inputs = [
    ProcessingInput(source=preprocess_s3_input, destination=cfg.container_input)
]
preprocess_outputs = [
    ProcessingOutput(
        source=cfg.container_train_output,
        destination=preprocess_s3_train_output,
    ),
    ProcessingOutput(
        source=cfg.container_validation_output,
        destination=preprocess_s3_validation_output,
    ),
    ProcessingOutput(
        source=cfg.container_test_output, destination=preprocess_s3_test_output
    ),
]

# Train model variables
train_hyperparameter = {
    "max_depth": 10,
    "eta": 0.1,
    "objective": "reg:squarederror",
    "num_round": 100,
    "colsample_bytree": 0.3,
    "alpha": 10,
    "data_format": content_type,
    "tracking_uri": cfg.tracking_uri,
    "experiment_name": "retail-sales",
}

train_s3_output = f"{cfg.model_bucket}/{folder}"
train_input = TrainingInput(preprocess_s3_train_output, content_type=content_type)
validation_input = TrainingInput(
    preprocess_s3_validation_output, content_type=content_type
)
test_input = TrainingInput(preprocess_s3_test_output, content_type=content_type)

train_data_channels = {
    "train": train_input,
    "validation": validation_input,
    "test": test_input,
}

xgb_estimator = XGBoost(
    entry_point="retail_train.py",
    source_dir=source_dir,
    framework_version="1.3-1",
    hyperparameters=train_hyperparameter,
    role=cfg.sagemaker_role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=train_s3_output,
    code_location=train_s3_output,
    base_job_name="retail-sales",
    subnets=cfg.subnet,
    security_group_ids=cfg.security_group_ids,
)

dag_args = {
    "start_date": days_ago(2),
    "catchup": False,
    "depends_on_past": False,
    "provide_context": True,
    "retries": 0,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="retail_sales",
    description="Retail prediction",
    concurrency=1,
    schedule_interval="@once",
    default_args=dag_args,
    tags=["retail", "sales", "prediction"],
) as dag:
    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_process_job,
        op_kwargs={
            "processor": sklearn_processor,
            "code": preprocess_code,
            "inputs": preprocess_inputs,
            "outputs": preprocess_outputs,
            "arguments": preprocess_arguments,
        },
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_train_job,
        op_kwargs={
            "estimator": xgb_estimator,
            "data_channels": train_data_channels,
        },
    )

    preprocess_data >> train_model
