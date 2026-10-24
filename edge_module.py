"""Edge inference module.

Receives smart-meter readings, predicts each home's electricity use for the coming hour
and stores the prediction for the dashboard.

Sources (where readings come from):
  http   (default) - the simulator POSTs readings to http://127.0.0.1:5000/telemetry
  iothub           - reads device-to-cloud messages from Azure IoT Hub's
                     Event Hub-compatible endpoint (set IOTHUB_EVENTHUB_CONNECTION_STRING)

Predictors (who runs the model):
  local    (default) - the model file on this device (edge inference)
  endpoint           - the model deployed as an Azure ML online endpoint
                       (set AZURE_ENDPOINT_URL and AZURE_ENDPOINT_KEY)

Usage:
  python edge_module.py
  python edge_module.py --source iothub
  python edge_module.py --predictor endpoint
"""
import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
import requests
from flask import Flask, jsonify, request

from config import (
    AZURE_ENDPOINT_KEY, AZURE_ENDPOINT_URL, EDGE_HOST, EDGE_PORT, IOTHUB_CONSUMER_GROUP,
    IOTHUB_EVENTHUB_CONNECTION_STRING, MODEL_PATH,
)
from features import FEATURES, load_level, make_features
from storage import connect, insert_prediction, recent_predictions


class LocalPredictor:
    name = "local"

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        print(f"[edge] model loaded from {MODEL_PATH}")

    def predict(self, reading):
        X = pd.DataFrame([make_features(reading)], columns=FEATURES)
        return float(self.model.predict(X)[0])


class EndpointPredictor:
    name = "endpoint"

    def __init__(self):
        if not (AZURE_ENDPOINT_URL and AZURE_ENDPOINT_KEY):
            raise SystemExit("Set AZURE_ENDPOINT_URL and AZURE_ENDPOINT_KEY in .env "
                             "(python azure_ml.py endpoint-info prints them)")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {AZURE_ENDPOINT_KEY}"
        print(f"[edge] using Azure ML endpoint {AZURE_ENDPOINT_URL}")

    def predict(self, reading):
        make_features(reading)  # validate before sending
        r = self.session.post(AZURE_ENDPOINT_URL, json={"data": [reading]}, timeout=10)
        r.raise_for_status()
        return float(r.json()["predictions"][0])


class EdgeModule:
    def __init__(self, predictor):
        self.predictor = predictor
        self.conn = connect()

    def process(self, reading, source):
        start = time.perf_counter()
        predicted = round(max(self.predictor.predict(reading), 0.0), 3)
        inference_ms = (time.perf_counter() - start) * 1000
        record = {
            "device_id": reading["device_id"],
            "timestamp": reading["timestamp"],
            "temperature": float(reading["temperature"]),
            "humidity": float(reading["humidity"]),
            "occupancy": int(reading["occupancy"]),
            "prev_kwh": float(reading["prev_kwh"]),
            "predicted_kwh": predicted,
            "load_level": load_level(predicted),
            "inference_ms": round(inference_ms, 3),
            "predictor": self.predictor.name,
            "source": source,
            "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        insert_prediction(self.conn, record)
        print(
            f"[edge] {record['device_id']:<9} {record['timestamp']}  T={record['temperature']:>4.1f}°C "
            f"occ={record['occupancy']}  last hour={record['prev_kwh']:.2f} kWh -> "
            f"next hour {predicted:.2f} kWh ({record['load_level']})"
        )
        return record


def create_app(edge):
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(status="ok", predictor=edge.predictor.name, model=MODEL_PATH.name)

    @app.post("/telemetry")
    def telemetry():
        reading = request.get_json(silent=True)
        if not isinstance(reading, dict):
            return jsonify(error="expected a JSON object"), 400
        try:
            record = edge.process(reading, source="http")
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        except requests.RequestException as exc:
            return jsonify(error=f"endpoint call failed: {exc}"), 502
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
        except (ValueError, TypeError, requests.RequestException) as exc:
            print(f"[edge] skipped message: {exc}")

    client = EventHubConsumerClient.from_connection_string(
        IOTHUB_EVENTHUB_CONNECTION_STRING, consumer_group=IOTHUB_CONSUMER_GROUP
    )
    print("[edge] listening to Azure IoT Hub ... (Ctrl+C to stop)")
    with client:
        client.receive(on_event=on_event, starting_position="@latest")


def main():
    parser = argparse.ArgumentParser(description="Energy edge inference module")
    parser.add_argument("--source", choices=["http", "iothub"], default="http")
    parser.add_argument("--predictor", choices=["local", "endpoint"], default="local")
    args = parser.parse_args()

    predictor = EndpointPredictor() if args.predictor == "endpoint" else LocalPredictor()
    edge = EdgeModule(predictor)
    if args.source == "iothub":
        run_iothub(edge)
    else:
        print(f"[edge] listening on http://{EDGE_HOST}:{EDGE_PORT}/telemetry")
        create_app(edge).run(host=EDGE_HOST, port=EDGE_PORT)


if __name__ == "__main__":
    main()
