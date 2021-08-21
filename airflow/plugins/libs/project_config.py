"""Project data configuration."""

sagemaker_role = "SageMaker-Studio"
sagemaker_jobs_dir = "/home/ubuntu/sagemaker_jobs"

raw_bucket = "s3://mlops-raw-data"
curated_bucket = "s3://mlops-curated-data"
model_bucket = "s3://mlops-model-artifacts"

container_input = "/opt/ml/processing/input"
container_train_output = "/opt/ml/processing/train"
container_validation_output = "/opt/ml/processing/validation"
container_test_output = "/opt/ml/processing/test"

tracking_uri = "http://10.0.6.196:5000"
subnet = ["subnet-ba6a47de"]
security_group_ids = ["sg-0a551f8a939e69e31"]
