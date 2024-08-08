from prefect import task, flow
from prefect.deployments import Deployment
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
)
from sklearn.feature_extraction import DictVectorizer

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import time
import xgboost as xgb
import pickle
import os

from prometheus_client import Summary, Counter, Gauge

# Prometheus metrics
REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")
TRAINING_COUNTER = Counter(
    "model_training_count", "Number of times the model has been trained"
)
TRAINING_DURATION = Gauge(
    "model_training_duration_seconds", "Duration of the last model training in seconds"
)


@task
@REQUEST_TIME.time()
def load_data():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_data_path = os.path.join(
        project_dir, "data", "train_timeseries", "train_timeseries.csv"
    )
    val_data_path = os.path.join(
        project_dir, "data", "validation_timeseries", "validation_timeseries.csv"
    )
    test_data_path = os.path.join(
        project_dir, "data", "test_timeseries", "test_timeseries.csv"
    )

    train_data = pd.read_csv(train_data_path, sep=",")
    test_data = pd.read_csv(test_data_path, sep=",")
    val_data = pd.read_csv(val_data_path, sep=",")

    return train_data, test_data, val_data


@task
@REQUEST_TIME.time()
def preprocess_data(train_data, test_data, val_data):
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    val_data = val_data.dropna()

    def fix_datetime_and_score(df):
        df["year"] = pd.DatetimeIndex(df["date"]).year
        df["month"] = pd.DatetimeIndex(df["date"]).month
        df["day"] = pd.DatetimeIndex(df["date"]).day
        df["score"] = df["score"].round().astype(int)
        return df

    train_data = fix_datetime_and_score(train_data)
    test_data = fix_datetime_and_score(test_data)
    val_data = fix_datetime_and_score(val_data)

    numerical_column_list = [
        "PRECTOT",
        "PS",
        "QV2M",
        "T2M",
        "T2MDEW",
        "T2M_MAX",
        "T2M_MIN",
        "T2M_RANGE",
        "WS10M",
        "WS10M_MAX",
        "WS10M_MIN",
        "WS10M_RANGE",
        "WS50M",
        "WS50M_MAX",
        "WS50M_MIN",
        "WS50M_RANGE",
    ]
    categorical_column_list = ["year", "month", "day"]

    def remove_outliers(df):
        for col_name in numerical_column_list:
            df = df[
                (df[col_name] <= df[col_name].mean() + 3 * df[col_name].std())
                & (df[col_name] >= df[col_name].mean() - 3 * df[col_name].std())
            ]
        return df

    train_data = remove_outliers(train_data)
    test_data = remove_outliers(test_data)
    val_data = remove_outliers(val_data)

    def remove_fips_and_date(df):
        df = df.drop("fips", axis=1)
        df = df.drop("date", axis=1)
        return df

    train_data = remove_fips_and_date(train_data)
    test_data = remove_fips_and_date(test_data)
    val_data = remove_fips_and_date(val_data)

    def remove_invariates(df):
        df = df.drop("TS", axis=1)
        df = df.drop("T2MWET", axis=1)
        return df

    train_data = remove_invariates(train_data)
    test_data = remove_invariates(test_data)
    val_data = remove_invariates(val_data)

    dv = DictVectorizer()

    df_train = train_data.drop("score", axis=1)
    train_dicts = df_train[categorical_column_list + numerical_column_list].to_dict(
        orient="records"
    )
    X_train = dv.fit_transform(train_dicts)
    y_train = train_data["score"]

    df_val = val_data.drop("score", axis=1)
    val_dicts = df_val[categorical_column_list + numerical_column_list].to_dict(
        orient="records"
    )
    X_val = dv.fit_transform(val_dicts)
    y_val = val_data["score"]

    df_test = test_data.drop("score", axis=1)
    test_dicts = df_test[categorical_column_list + numerical_column_list].to_dict(
        orient="records"
    )
    X_test = dv.fit_transform(test_dicts)
    y_test = test_data["score"]

    return X_train, y_train, X_val, y_val, X_test, y_test, dv


@task
@REQUEST_TIME.time()
@TRAINING_COUNTER.count_exceptions()
def train_and_log_model(X_train, y_train, X_val, y_val, X_test, y_test, dv):
    # Define the model
    # model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=12)
    model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=12)
    # model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=12)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("drought-prediction-experiment")

    # Start an MLflow run
    with mlflow.start_run() as run:
        mlflow.set_tag("developer", "Ibai")
        mlflow.set_tag("model", "xgboost")

        start_time = time.time()

        # Train the model
        model.fit(X_train, y_train)

        end_time = time.time()
        training_duration = end_time - start_time
        TRAINING_DURATION.set(training_duration)

        # Make predictions on the validation set
        val_predictions = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_predictions)
        val_accuracy = accuracy_score(y_val, val_predictions)

        # Make predictions on the test set
        test_predictions = model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("training_duration", training_duration)

        # Save the model locally
        model_path = "models/model_xgboost.bin"
        dv_path = "models/dv.bin"
        with open(model_path, "wb") as f_out:
            pickle.dump(model, f_out)
        with open(dv_path, "wb") as f_out:
            pickle.dump(dv, f_out)

        # Log the model
        mlflow.xgboost.log_model(model, "model")
        mlflow.log_artifact(local_path=model_path, artifact_path="models_pickle")
        mlflow.log_artifact(local_path=dv_path, artifact_path="models_pickle")

        # Register the model in the MLflow Model Registry
        model_uri = f"runs: /{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, "drought-prediction-model")

        return registered_model.version


# Define a task to promote the model to staging
@task
@REQUEST_TIME.time()
def promote_model_to_staging(model_version):
    from mlflow.tracking import MlflowClient

    model_name = "drought-prediction-model"
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Promote the model to "Staging"
    client.transition_model_version_stage(
        name=model_name, version=model_version, stage="Staging"
    )

    print(
        f"Model version {model_version} of '{model_name}' has been promoted to 'Staging'."
    )


# Define the Prefect flow
@flow
def mlflow_workflow():
    # Load data
    (
        train_data,
        test_data,
        val_data,
    ) = load_data()

    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, dv = preprocess_data(
        train_data, test_data, val_data
    )

    # Train and log model
    model_version = train_and_log_model(
        X_train, y_train, X_val, y_val, X_test, y_test, dv
    )

    # Promote model to staging
    promote_model_to_staging(model_version)


if __name__ == "__main__":
    # Register the flow
    deployment = Deployment.build_from_flow(
        flow=mlflow_workflow, name="Draught_prediction"
    )
    deployment.apply()

    # Run the flow
    mlflow_workflow()
