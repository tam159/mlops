# ML Ops: Machine Learning Operations
This repository contains some examples of MLOps with code

## Machine learning workflow with Airflow, MLflow and SageMaker
![ML Workflow](https://raw.githubusercontent.com/tam159/mlops/main/images/mlworkflow.png)

### Workflow tasks are scheduled by Airflow and experiments are logged in MLflow

0. The business problem is framed as a machine learning problem: what is observed and what is predicted.
1. Data acquisition: ingesting data from sources including data collection, data integration and data quality checking.
2. Data pre-processing: handling missing data, outliers, long tails, .etc.
3. Feature engineering: running experiments with different features, adding, removing and changing features.
4. Data transformation: standardizing data, converting data format compatible with training algorithms.
5. Job training: training’s parameters, metrics, .etc are tracked in the MLflow. We can also run SageMaker Hyperparameter Optimization with many training jobs then search the metrics and params in the MLflow for a comparison with minimal effort to find the best version of a model.
6. Model evaluation: analyzing model performance based on predicted results on test data.
7. If business goals are met, the model will be registered in the SageMaker Inference Models. We can also register the model in the MLflow.
8. Getting predictions in any of the following ways:
   1. Using SageMaker Batch Transform to get predictions for an entire dataset.
   2. Setting up a persistent endpoint to get one prediction at a time using SageMaker Inference Endpoints.
9. Monitoring and debugging the workflow, re-training with a data augmentation.

#### For the data processing, feature engineering and model evaluation, we can use several AWS services
- EMR: providing a Hadoop ecosystem cluster including pre-installed Spark, Flink, .etc. We should use a transient cluster to process the data and terminate it when all done.
- Glue job: providing a server-less Apache Spark, Python environments. Glue’ve supported Spark 3.1 since 2021 Aug.
- SageMaker Processing jobs: running in containers, there are many prebuilt images supporting data science. It also supports Spark 3.

#### For other steps, we can use AWS SageMaker for job training, hyperparameter tuning, model serving and production monitoring

#### Airflow and MLflow store their metadata in AWS RDS for PostgreSQL

### Data accessing
- All data stored in S3 can be queried via Athena with metadata from Glue data catalog.
- We can also ingest the data into SageMaker Feature Store in batches directly to the offline store.

### [Medium link][ML workflow Medium]


<!-- links -->
[ML workflow Medium]: https://tam159.medium.com/ml-workflow-with-airflow-mlflow-and-sagemaker-ad076e5f614b
