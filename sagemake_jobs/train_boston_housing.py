import sagemaker
# import pandas as pd
# from sklearn.datasets import load_boston
from sagemaker.sklearn.estimator import SKLearn
# from sklearn.model_selection import train_test_split

sess = sagemaker.Session()
role = "SageMaker-Studio"
# bucket = sess.default_bucket()

# uri of your remote mlflow server
tracking_uri = "http://52.76.38.6:5000"

train_path = "s3://mlops-curated-data/boston_housing/preprocess/train/"
test_path = "s3://mlops-curated-data/boston_housing/preprocess/test/"
output_path = "s3://mlops-model-artifacts/boston_housing/artifacts/"

hyperparameters = {
    'tracking_uri': tracking_uri,
    'experiment_name': 'boston-housing',
    'n-estimators': 100,
    'min-samples-leaf': 3,
    'features': 'CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT',
    'target': 'target',
    'train-file': 'boston_train.csv',
    'test-file': 'boston_test.csv'
}

metric_definitions = [{'Name': 'median-AE', 'Regex': "AE-at-50th-percentile: ([0-9.]+).*$"}]

estimator = SKLearn(
    entry_point='train.py',
    source_dir='boston_housing',
    role=role,
    metric_definitions=metric_definitions,
    hyperparameters=hyperparameters,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    framework_version='0.23-1',
    base_job_name='boston-housing-mlflow',
    output_path=output_path,
)

estimator.fit({'train': train_path, 'test': test_path})
