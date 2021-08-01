"""Example of Sagemaker training job with sklearn."""

import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import HyperparameterTuner, IntegerParameter

sess = sagemaker.Session()
role = "SageMaker-Studio"

tracking_uri = "http://52.76.38.6:5000"

train_path = "s3://mlops-curated-data/boston_housing/preprocess/train/boston_train.csv"
test_path = "s3://mlops-curated-data/boston_housing/preprocess/test/boston_test.csv"

hyperparameters = {
    "tracking_uri": tracking_uri,
    "experiment_name": "boston-housing",
    "features": "CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT",
    "target": "target",
}

metric_definitions = [
    {"Name": "median-AE", "Regex": "AE-at-50th-percentile: ([0-9.]+).*$"}
]


estimator = SKLearn(
    entry_point="boston_housing.py",
    source_dir="train",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    hyperparameters=hyperparameters,
    metric_definitions=metric_definitions,
    framework_version="0.23-1",
    py_version="py3",
)

hyperparameter_ranges = {
    "n-estimators": IntegerParameter(50, 200),
    "min-samples-leaf": IntegerParameter(1, 10),
}

objective_metric_name = "median-AE"
objective_type = "Minimize"


tuner = HyperparameterTuner(
    estimator,
    objective_metric_name,
    hyperparameter_ranges,
    metric_definitions,
    max_jobs=4,
    max_parallel_jobs=2,
    objective_type=objective_type,
    base_tuning_job_name="tune-sklearn-mlflow",
)

tuner.fit({"train": train_path, "test": test_path})
