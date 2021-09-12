"""Tuning retail sales forecast with XGBoost."""

from sagemaker.inputs import TrainingInput
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter
from sagemaker.xgboost.estimator import XGBoost

tracking_uri = "http://10.0.6.196:5000"
role = "SageMaker-Studio"
script_path = "retail_train.py"
source_dir = "source_dir"

train_path = "s3://mlops-curated-data/retail_sales/preprocess/train"
validation_path = "s3://mlops-curated-data/retail_sales/preprocess/validation"
test_path = "s3://mlops-curated-data/retail_sales/preprocess/test"
output_path = "s3://mlops-model-artifacts/retail_sales"

instance_type = "ml.m5.xlarge"
content_type = "csv"

hyperparams = {
    "data_format": content_type,
    "tracking_uri": tracking_uri,
    "experiment_name": "retail-sales",
}

xgb_script_mode_estimator = XGBoost(
    entry_point=script_path,
    source_dir=source_dir,
    framework_version="1.3-1",
    hyperparameters=hyperparams,
    role=role,
    instance_count=1,
    instance_type=instance_type,
    output_path=output_path,
    code_location=output_path,
    base_job_name="retail-sales",
    subnets=["subnet-ba6a47de"],
    security_group_ids=["sg-0a551f8a939e69e31"],
)

train_input = TrainingInput(train_path, content_type=content_type)
validation_input = TrainingInput(validation_path, content_type=content_type)
test_input = TrainingInput(test_path, content_type=content_type)

data_channels = {
    "train": train_input,
    "validation": validation_input,
    "test": test_input,
}

hyperparameter_ranges = {
    "num_round": IntegerParameter(50, 150),
    "max_depth": IntegerParameter(5, 15),
    "eta": ContinuousParameter(0.01, 0.5),
    "colsample_bytree": ContinuousParameter(0.01, 0.5),
    "alpha": ContinuousParameter(1, 20),
}

tuner = HyperparameterTuner(
    estimator=xgb_script_mode_estimator,
    objective_metric_name="validation:rmse",
    hyperparameter_ranges=hyperparameter_ranges,
    strategy="Bayesian",
    objective_type="Minimize",
    max_jobs=20,
    max_parallel_jobs=4,
    base_tuning_job_name="tune-retail-sales",
)

tuner.fit(data_channels)
