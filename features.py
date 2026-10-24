"""Feature engineering shared by training, the edge module and the Azure ML endpoint.

A smart-meter reading arrives as:
  {"device_id", "timestamp", "temperature", "humidity", "occupancy", "prev_kwh"}
where prev_kwh is the energy used in the hour that just ended. The model predicts the
energy (kWh) the home will use in the coming hour.
"""
from datetime import datetime

FEATURES = ["hour", "day_of_week", "is_weekend", "temperature", "humidity", "occupancy", "prev_kwh"]
READING_FIELDS = ["device_id", "timestamp", "temperature", "humidity", "occupancy", "prev_kwh"]

# Predicted kWh for the coming hour -> load level shown on the dashboard
LOAD_LEVELS = [(0.6, "Low"), (1.5, "Normal"), (2.5, "High"), (float("inf"), "Peak")]


def make_features(reading):
    """Turn a raw meter reading into the model's input row (time features derived at the edge)."""
    missing = [f for f in READING_FIELDS if f not in reading]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    ts = datetime.fromisoformat(str(reading["timestamp"]))
    return {
        "hour": ts.hour,
        "day_of_week": ts.weekday(),
        "is_weekend": int(ts.weekday() >= 5),
        "temperature": float(reading["temperature"]),
        "humidity": float(reading["humidity"]),
        "occupancy": int(reading["occupancy"]),
        "prev_kwh": float(reading["prev_kwh"]),
    }


def load_level(kwh):
    return next(label for limit, label in LOAD_LEVELS if kwh < limit)
