"""Scoring script for the Azure ML managed online endpoint.

Request:  {"data": [{"device_id", "timestamp", "temperature", "humidity", "occupancy", "prev_kwh"}, ...]}
Response: {"predictions": [kWh for the coming hour, ...], "load_levels": [...]}
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURES, load_level, make_features

model = None


def init():
    global model
    model_dir = Path(os.environ["AZUREML_MODEL_DIR"])
    model = joblib.load(next(model_dir.rglob("*.pkl")))


def run(raw_data):
    try:
        readings = json.loads(raw_data)["data"]
        X = pd.DataFrame([make_features(r) for r in readings], columns=FEATURES)
        predictions = [round(max(float(p), 0.0), 3) for p in model.predict(X)]
        return {"predictions": predictions, "load_levels": [load_level(p) for p in predictions]}
    except (KeyError, ValueError, TypeError) as exc:
        return {"error": str(exc)}
