# ML Ops: Machine Learning Operations
This repository contains some examples of MLOps with code

## Machine learning workflow with Airflow, MLflow and SageMaker
![MLOps Architecture](https://raw.githubusercontent.com/tam159/mlops/main/MLOps.png)

#### Airflow and MLflow store their metadata in AWS RDS for PostgreSQL

### All tasks are scheduled by Airflow. All experiments are logged in MLflow
1. Data pre-processing: handling missing data, outliers, long tails, etc
2. Feature engineering: running experiments with different features, adding, removing and changing features
3. Data transformation: standardizing data, converting data format compatible with training algorithms
4. Model training's parameters, metrics, are logged in MLflow. The model is registered in SageMaker Inference Model
5. Batch inference: performing batch transform on test data set
6. Model evaluation: analyzing model performance based on predicted results on test data
7. Model deployment: creating a SageMaker Inference Endpoint

#### For the first 3 steps, we can use some AWS services
- EMR: providing a Hadoop ecosystem cluster including Spark, Flink, etc
- Glue job: providing a server-less Apache Spark, Python environments 
- Sagemaker Processing jobs: running in containers, there are many prebuilt images supporting data science

#### For other steps, we can use AWS SageMaker for training jobs, batch transform jobs and endpoint deployments
