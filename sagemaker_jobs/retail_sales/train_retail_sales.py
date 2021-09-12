"""Training retail sales forecast with XGBoost."""

from sagemaker.inputs import TrainingInput
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
    "max_depth": 10,
    "eta": 0.1,
    "objective": "reg:squarederror",
    "num_round": 20,
    "colsample_bytree": 0.3,
    "alpha": 10,
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

xgb_script_mode_estimator.fit(data_channels)
