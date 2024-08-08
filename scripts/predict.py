import pandas as pd
import pickle
import os
from flask import Flask, request, jsonify
from prometheus_client import (
    Counter,
    Gauge,
    generate_latest,
)

app = Flask(__name__)

model_path = os.getenv("MODEL_PATH", "models/model_xgboost.bin")
dv_path = os.getenv("DV_PATH", "models/dv.bin")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "prediction_request_count", "Number of prediction requests received"
)
PREDICTION_DURATION = Gauge(
    "prediction_duration_seconds", "Time spent making a prediction"
)


def load_model_and_dv(model_path, dv_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(dv_path, "rb") as f:
        dv = pickle.load(f)
    return model, dv


def preprocess_input_data(input_data, dv):
    # Preprocess input data similarly to training data
    input_data = input_data.dropna()

    input_data["year"] = pd.DatetimeIndex(input_data["date"]).year
    input_data["month"] = pd.DatetimeIndex(input_data["date"]).month
    input_data["day"] = pd.DatetimeIndex(input_data["date"]).day

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

    input_data = input_data.drop("fips", axis=1)
    input_data = input_data.drop("date", axis=1)
    input_data = input_data.drop("TS", axis=1)
    input_data = input_data.drop("T2MWET", axis=1)

    input_dicts = input_data[categorical_column_list + numerical_column_list].to_dict(
        orient="records"
    )
    X_input = dv.transform(input_dicts)

    return X_input


@app.route("/predict", methods=["POST"])
# def predict(input_data, model_path=model_path, dv_path=dv_path):
def predict():
    input_data = request.json
    input_data = pd.DataFrame(input_data)

    # Increment the request count
    REQUEST_COUNT.inc()

    # Load the trained model and DictVectorizer
    model, dv = load_model_and_dv(model_path, dv_path)

    # Load and preprocess input data
    X_input = preprocess_input_data(input_data, dv)

    # Measure prediction duration
    with PREDICTION_DURATION.time():
        # Make predictions
        predictions = model.predict(X_input)

    # return predictions
    return jsonify({"predictions": predictions.tolist()})


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200


if __name__ == "__main__":
    # input_data_path = 'path_to_input_data.csv'
    # input_data = create_random_example()
    # predictions = predict(input_data)
    # print(f"The Drought score is: {predictions[0]}")
    app.run(host="0.0.0.0", port=5002)
