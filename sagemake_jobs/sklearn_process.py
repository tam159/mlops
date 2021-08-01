"""Example of Sagemaker processing with sklearn."""

import boto3
import pandas as pd
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

region = boto3.session.Session().region_name

role = "SageMaker-Studio"
sklearn_processor = SKLearnProcessor(
    framework_version="0.20.0", role=role, instance_type="ml.m5.large", instance_count=1
)

input_data = "s3://sagemaker-sample-data-{}/processing/census/census-income.csv".format(
    region
)
output_data = "s3://mlops-curated-data/census_income/preprocess"

df = pd.read_csv(input_data, nrows=10)
df.head(n=10)

sklearn_processor.run(
    code="preprocessing.py",
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/train", destination=f"{output_data}/train"
        ),
        ProcessingOutput(
            source="/opt/ml/processing/test", destination=f"{output_data}/test"
        ),
    ],
    arguments=["--train_test_split_ratio", "0.2"],
)

preprocessing_job_description = sklearn_processor.jobs[-1].describe()

output_config = preprocessing_job_description["ProcessingOutputConfig"]
for output in output_config["Outputs"]:
    if output["OutputName"] == "train_data":
        preprocessed_training_data = output["S3Output"]["S3Uri"]
    if output["OutputName"] == "test_data":
        preprocessed_test_data = output["S3Output"]["S3Uri"]
