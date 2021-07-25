"""Data parameter configuration."""

from sagemaker.tuner import ContinuousParameter

config = {
    "job_level": {"region_name": "ap-southeast-1", "run_hyperparameter_opt": "no"},
    "preprocess_data": {
        "s3_in_url": "s3://mlops-raw-data/video-reviews/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz",
        "s3_out_bucket": "mlops-curated-data",
        "s3_out_prefix": "video-reviews/preprocess",
        "delimiter": "\t",
    },
    "prepare_data": {
        "s3_in_bucket": "mlops-curated-data",
        "s3_in_prefix": "video-reviews/preprocess",
        "s3_out_bucket": "mlops-curated-data",
        "s3_out_prefix": "video-reviews/prepare",
    },
    "train_model": {
        "sagemaker_role": "AirflowEC2Role",
        "estimator_config": {
            "train_instance_count": 1,
            "train_instance_type": "ml.c5.4xlarge",
            "train_volume_size": 30,
            "train_max_run": 3600,
            "output_path": "s3://mlops-model-artifacts/video-reviews/",
            "base_job_name": "trng-recommender",
            "hyperparameters": {
                "feature_dim": "178729",
                "epochs": "10",
                "mini_batch_size": "200",
                "num_factors": "64",
                "predictor_type": "regressor",
            },
        },
        "inputs": {
            "train": "s3://mlops-curated-data/video-reviews/prepare/train/train.protobuf",
        },
    },
    "tune_model": {
        "tuner_config": {
            "objective_metric_name": "test:rmse",
            "objective_type": "Minimize",
            "hyperparameter_ranges": {
                "factors_lr": ContinuousParameter(0.0001, 0.2),
                "factors_init_sigma": ContinuousParameter(0.0001, 1),
            },
            "max_jobs": 20,
            "max_parallel_jobs": 2,
            "base_tuning_job_name": "hpo-recommender",
        },
        "inputs": {
            "train": "s3://mlops-curated-data/video-reviews/prepare/train/train.protobuf",
            "test": "s3://mlops-curated-data/video-reviews/prepare/validate/validate.protobuf",
        },
    },
    "batch_transform": {
        "transform_config": {
            "instance_count": 1,
            "instance_type": "ml.c4.xlarge",
            "data": "s3://mlops-curated-data/video-reviews/prepare/test/",
            "data_type": "S3Prefix",
            "content_type": "application/x-recordio-protobuf",
            "strategy": "MultiRecord",
            "output_path": "s3://mlops-curated-data/video-reviews/transform/",
        }
    },
}
