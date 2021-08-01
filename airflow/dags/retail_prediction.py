"""Airflow DAG for retail prediction."""

from datetime import timedelta

import boto3
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from libs import project_config as cfg
from libs.sagemaker_libs import run_process_job
from sagemaker.sklearn.processing import SKLearnProcessor

region = boto3.session.Session().region_name

role = cfg.sagemaker_role
curated_bucket = cfg.curated_bucket

job_folder = f"{cfg.sagemaker_jobs_dir}/retail_prediction"
code = f"{job_folder}/preprocess.py"
input_path = f"s3://sagemaker-sample-data-{region}/processing/census/census-income.csv"
output_path = f"{curated_bucket}/census_income/preprocess"
process_arguments = ["--train_test_split_ratio", "0.2"]

sklearn_processor = SKLearnProcessor(
    framework_version="0.20.0", role=role, instance_type="ml.m5.large", instance_count=1
)

default_args = {
    "start_date": days_ago(2),
    "catchup": False,
    "depends_on_past": False,
    "provide_context": True,
    "retries": 0,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="retail_prediction",
    description="Retail prediction",
    concurrency=1,
    schedule_interval="@once",
    default_args=default_args,
    tags=["retail", "prediction"],
) as dag:
    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_process_job,
        op_kwargs={
            "processor": sklearn_processor,
            "code": code,
            "input_path": input_path,
            "output_path": output_path,
            "arguments": process_arguments,
        },
    )
