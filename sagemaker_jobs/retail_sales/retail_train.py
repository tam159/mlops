"""Training entry point."""

import argparse
import logging
import os
import pickle as pkl

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="reg:linear")
    parser.add_argument("--num_round", type=int)
    parser.add_argument("--colsample_bytree", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--data_format", type=str)

    # MLflow related parameters
    parser.add_argument("--tracking_uri", type=str)
    parser.add_argument("--experiment_name", type=str)

    # SageMaker specific arguments. Defaults are set in the environment variables
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument(
        "--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"]
    )
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args = parser.parse_args()

    # Set remote mlflow server
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        train_hp = {
            "max_depth": args.max_depth,
            "eta": args.eta,
            "objective": args.objective,
            "num_round": args.num_round,
            "colsample_bytree": args.colsample_bytree,
            "alpha": args.alpha,
        }

        mlflow.log_params(train_hp)

        train_data = pd.read_csv(f"{args.train}/train.csv")
        X_train = train_data.iloc[:, 1:]
        y_train = train_data.iloc[:, 0]

        validation_data = pd.read_csv(f"{args.validation}/validation.csv")
        X_validation = validation_data.iloc[:, 1:]
        y_validation = validation_data.iloc[:, 0]

        test_data = pd.read_csv(f"{args.test}/test.csv")
        X_test = test_data.iloc[:, 1:]
        y_test = test_data.iloc[:, 0]

        d_train = xgb.DMatrix(data=X_train, label=y_train)
        d_val = xgb.DMatrix(data=X_validation, label=y_validation)
        d_test = xgb.DMatrix(data=X_test, label=y_test)

        watchlist = (
            [(d_train, "train"), (d_val, "validation")]
            if d_val is not None
            else [(d_train, "train")]
        )

        bst = xgb.train(
            params=train_hp,
            dtrain=d_train,
            evals=watchlist,
            num_boost_round=args.num_round,
        )

        # Evaluate model
        predicted_values = bst.predict(d_test)

        RMSE = float(
            format(np.sqrt(mean_squared_error(y_test, predicted_values)), ".3f")
        )
        MSE = mean_squared_error(y_test, predicted_values)
        MAE = mean_absolute_error(y_test, predicted_values)
        r2 = r2_score(y_test, predicted_values)
        adj_r2 = 1 - (1 - r2) * (len(X_test) - 1) / (len(X_test) - X_test.shape[1] - 1)

        # Log metrics
        mlflow.log_metrics(
            {"RMSE": RMSE, "MSE": MSE, "MAE": MAE, "r2": r2, "adj_r2": adj_r2}
        )

        # Save MLflow Model
        logging.info("saving model in MLflow")
        mlflow.xgboost.log_model(bst, "model")

    # Save the model to the location specified by model_dir
    model_location = f"{args.model_dir}/xgboost-model"
    with open(model_location, "wb") as handle:
        pkl.dump(bst, handle)
    logging.info(f"Stored trained model at {model_location}")
