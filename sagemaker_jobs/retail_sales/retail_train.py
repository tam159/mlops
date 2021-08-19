"""Training entry point."""

import argparse
import logging
import os
import pickle as pkl

import mlflow
import xgboost as xgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="reg:linear")
    parser.add_argument("--num_round", type=int)
    parser.add_argument("--colsample_bytree", type=float)
    parser.add_argument("--alpha", type=int)
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

        dtrain = xgb.DMatrix(f"{args.train}?format={args.data_format}")
        dval = xgb.DMatrix(f"{args.validation}?format={args.data_format}")
        watchlist = (
            [(dtrain, "train"), (dval, "validation")]
            if dval is not None
            else [(dtrain, "train")]
        )

        bst = xgb.train(
            params=train_hp,
            dtrain=dtrain,
            evals=watchlist,
            num_boost_round=args.num_round,
        )

        # Save MLflow Model
        logging.info("saving model in MLflow")
        mlflow.xgboost.log_model(bst, "model")

    # Save the model to the location specified by ``model_dir``
    model_location = f"{args.model_dir}/xgboost-model"
    with open(model_location, "wb") as handle:
        pkl.dump(bst, handle)
    logging.info(f"Stored trained model at {model_location}")
