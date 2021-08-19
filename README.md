# ML Ops: Machine Learning Operations
This repository contains some examples of MLOps with code

## Machine learning workflow with Airflow, MLflow and SageMaker
![MLOps Architecture](https://raw.githubusercontent.com/tam159/mlops/main/ml_workflow.png)

#### Airflow and MLflow store their metadata in AWS RDS for PostgreSQL

### Workflow tasks are scheduled by Airflow and experiments are logged in MLflow

0. The business problem is framed as a machine learning problem: what is observed and what should be predicted
1. Data acquisition: acquiring raw data from data sources including data collection and data integration
2. Data pre-processing: handling missing data, outliers, long tails, etc
3. Feature engineering: running experiments with different features, adding, removing and changing features
4. Data transformation: standardizing data, converting data format compatible with training algorithms
5. Job training: training's parameters, metrics, are logged in MLflow
6. Model evaluation: analyzing model performance based on predicted results on test data
7. If business goals are met, the model will be registered in the SageMaker Inference Models
8. Getting predictions in any of the following ways
   1. Using SageMaker Batch Transform to get predictions for an entire dataset
   2. Setting up a persistent endpoint to get one prediction at a time using SageMaker Inference Endpoints
9. Monitoring and debugging the workflow, re-training with a data augmentation

#### For the first 4 steps, we can use some AWS services
- EMR: providing a Hadoop ecosystem cluster including Spark, Flink, etc
- Glue job: providing a server-less Apache Spark, Python environments
- Sagemaker Processing jobs: running in containers, there are many prebuilt images supporting data science

#### For other steps, we can use AWS SageMaker for job training, model evaluation, deployment and prediction
