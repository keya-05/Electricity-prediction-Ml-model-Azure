"""Edge inference module.

Receives weather telemetry, runs the trained Decision Tree locally ("at the edge")
and stores each prediction for the dashboard.

Sources:
  http   (default) - the simulator POSTs readings to http://127.0.0.1:5000/telemetry
  iothub           - reads device-to-cloud messages from Azure IoT Hub's
                     Event Hub-compatible endpoint (set IOTHUB_EVENTHUB_CONNECTION_STRING)

Usage:
  python edge_module.py
  python edge_module.py --source iothub
"""
import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from config import (
    EDGE_HOST, EDGE_PORT, FEATURES, IOTHUB_CONSUMER_GROUP,
    IOTHUB_EVENTHUB_CONNECTION_STRING, MODEL_PATH, RAIN_LABEL,
)
from storage import connect, insert_prediction, recent_predictions


class WeatherPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.rain_index = list(self.model.classes_).index(RAIN_LABEL)

    def predict(self, reading):
        X = pd.DataFrame([[float(reading[f]) for f in FEATURES]], columns=FEATURES)
        start = time.perf_counter()
        label = self.model.predict(X)[0]
        rain_probability = float(self.model.predict_proba(X)[0][self.rain_index])
        inference_ms = (time.perf_counter() - start) * 1000
        return label, rain_probability, inference_ms


def validate(reading):
    missing = [f for f in FEATURES + ["device_id"] if f not in reading]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    for f in FEATURES:
        float(reading[f])  # raises ValueError for non-numeric values


class EdgeModule:
    def __init__(self):
        self.predictor = WeatherPredictor(MODEL_PATH)
        self.conn = connect()
        print(f"[edge] model loaded from {MODEL_PATH}")

    def process(self, reading, source):
        validate(reading)
        label, rain_probability, inference_ms = self.predictor.predict(reading)
        record = {
            "device_id": reading["device_id"],
            "timestamp": reading.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            **{f: float(reading[f]) for f in FEATURES},
            "prediction": label,
            "rain_probability": round(rain_probability, 3),
            "inference_ms": round(inference_ms, 3),
            "source": source,
        }
        insert_prediction(self.conn, record)
        print(
            f"[edge] {record['device_id']:<12} T={record['temperature']:>5.1f}°C "
            f"H={record['humidity']:>5.1f}% P={record['pressure']:>7.1f}hPa "
            f"W={record['wind_speed']:>5.1f}km/h -> {label} ({rain_probability:.0%})"
        )
        return record


def create_app(edge):
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(status="ok", model=str(MODEL_PATH.name))

    @app.post("/telemetry")
    def telemetry():
        reading = request.get_json(silent=True)
        if not isinstance(reading, dict):
            return jsonify(error="expected a JSON object"), 400
        try:
            record = edge.process(reading, source="http")
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(record)

    @app.get("/predictions")
    def predictions():
        limit = min(int(request.args.get("limit", 50)), 1000)
        return jsonify(recent_predictions(edge.conn, limit))

    return app


def run_iothub(edge):
    if not IOTHUB_EVENTHUB_CONNECTION_STRING:
        raise SystemExit("Set IOTHUB_EVENTHUB_CONNECTION_STRING in .env to use --source iothub")
    from azure.eventhub import EventHubConsumerClient

    def on_event(partition_context, event):
        try:
            edge.process(json.loads(event.body_as_str()), source="iothub")
        except (ValueError, TypeError) as exc:
            print(f"[edge] skipped bad message: {exc}")

    client = EventHubConsumerClient.from_connection_string(
        IOTHUB_EVENTHUB_CONNECTION_STRING, consumer_group=IOTHUB_CONSUMER_GROUP
    )
    print("[edge] listening to Azure IoT Hub ... (Ctrl+C to stop)")
    with client:
        client.receive(on_event=on_event, starting_position="@latest")


def main():
    parser = argparse.ArgumentParser(description="Weather edge inference module")
    parser.add_argument("--source", choices=["http", "iothub"], default="http")
    args = parser.parse_args()

    edge = EdgeModule()
    if args.source == "iothub":
        run_iothub(edge)
    else:
        print(f"[edge] listening on http://{EDGE_HOST}:{EDGE_PORT}/telemetry")
        create_app(edge).run(host=EDGE_HOST, port=EDGE_PORT)


if __name__ == "__main__":
    main()
