"""Example of Sagemaker training job with sklearn."""

import sagemaker
from sagemaker.sklearn.estimator import SKLearn

sess = sagemaker.Session()
role = "SageMaker-Studio"

tracking_uri = "http://10.0.6.196:5000"
source_dir = "source_dir"

train_path = "s3://mlops-curated-data/boston_housing/preprocess/train/"
test_path = "s3://mlops-curated-data/boston_housing/preprocess/test/"
output_path = "s3://mlops-model-artifacts/boston_housing/artifacts/"

hyperparameters = {
    "tracking_uri": tracking_uri,
    "experiment_name": "boston-housing",
    "n-estimators": 100,
    "min-samples-leaf": 3,
    "features": "CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT",
    "target": "target",
    "train-file": "boston_train.csv",
    "test-file": "boston_test.csv",
}

metric_definitions = [
    {"Name": "median-AE", "Regex": "AE-at-50th-percentile: ([0-9.]+).*$"}
]

estimator = SKLearn(
    entry_point="boston_train.py",
    source_dir=source_dir,
    role=role,
    metric_definitions=metric_definitions,
    hyperparameters=hyperparameters,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="0.23-1",
    base_job_name="boston-housing-mlflow",
    output_path=output_path,
)

estimator.fit({"train": train_path, "test": test_path})
