"""Retail sales data pre-processing."""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def get_month(x):
    """Get month from date."""
    return int(str(x).split("-")[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--validation_ratio", type=float, default=0.2)

    args = parser.parse_args()
    train_ratio = args.train_ratio
    validation_ratio = args.validation_ratio

    print("Received arguments {}".format(args))

    feature_input_data_path = os.path.join(
        "/opt/ml/processing/input", "Features_data_set.csv"
    )
    sales_input_data_path = os.path.join(
        "/opt/ml/processing/input", "sales_data_set.csv"
    )
    stores_input_data_path = os.path.join(
        "/opt/ml/processing/input", "stores_data_set.csv"
    )

    # Read from inputs
    feature = pd.read_csv(feature_input_data_path)
    sales = pd.read_csv(sales_input_data_path)
    stores = pd.read_csv(stores_input_data_path)

    # Change the datatype of 'date' column
    feature["Date"] = pd.to_datetime(feature["Date"])
    sales["Date"] = pd.to_datetime(sales["Date"])

    # MERGE DATASET INTO ONE DATAFRAME¶
    df = pd.merge(sales, feature, on=["Store", "Date", "IsHoliday"])
    df = pd.merge(df, stores, on=["Store"], how="left")

    # Add month
    df["month"] = df["Date"].apply(get_month)

    # Fill up NaN elements with zeros
    df = df.fillna(0)

    # Replace the "IsHoliday" with ones and zeros instead of True and False
    df["IsHoliday"] = df["IsHoliday"].replace({True: 1, False: 0})

    # Drop date columns
    df = df.drop(columns=["Date"])

    # Get dummies
    df = pd.get_dummies(df, columns=["Type", "Store", "Dept"], drop_first=True)

    # Move target to the first column
    df = pd.concat([df["Weekly_Sales"], df.drop(["Weekly_Sales"], axis=1)], axis=1)

    print(
        "Splitting data into train, validation and test sets with ratio {} - {} - {}".format(
            train_ratio, validation_ratio, 1 - train_ratio - validation_ratio
        )
    )
    train_data, validation_data, test_data = np.split(
        df.sample(frac=1, random_state=1234),
        [int(train_ratio * len(df)), int((train_ratio + validation_ratio) * len(df))],
    )

    train_data_output_path = os.path.join("/opt/ml/processing/train", "train.csv")
    validation_data_output_path = os.path.join(
        "/opt/ml/processing/validation", "validation.csv"
    )
    test_data_output_path = os.path.join("/opt/ml/processing/test", "test.csv")

    print("Saving train data to {}".format(train_data_output_path))
    pd.DataFrame(train_data).to_csv(train_data_output_path, header=True, index=False)

    print("Saving validation data to {}".format(validation_data_output_path))
    pd.DataFrame(validation_data).to_csv(
        validation_data_output_path, header=True, index=False
    )

    print("Saving test data to {}".format(test_data_output_path))
    pd.DataFrame(test_data).to_csv(test_data_output_path, header=True, index=False)
