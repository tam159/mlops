"""Project data configuration."""

sagemaker_role = "SageMaker-Studio"
sagemaker_jobs_dir = "/home/ubuntu/sagemaker_jobs"

raw_bucket = "s3://mlops-raw-data"
curated_bucket = "s3://mlops-curated-data"
model_bucket = "s3://mlops-model-artifacts"

container_input = "/opt/ml/processing/input"
container_train_output = "/opt/ml/processing/train"
container_test_output = "/opt/ml/processing/test"
