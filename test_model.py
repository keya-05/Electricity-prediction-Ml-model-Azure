import joblib
import pandas as pd

from features import FEATURES, load_level, make_features

# Load the trained model
model = joblib.load("energy_model.pkl")

# Sample smart-meter readings at the start of an hour
samples = [
    # Case 1: family home, weekday 19:00, hot evening, everyone home
    {"device_id": "home-101", "timestamp": "2026-10-21T19:00", "temperature": 31.0, "humidity": 62, "occupancy": 4, "prev_kwh": 2.4},
    # Case 2: same home, 03:00, cool night, everyone asleep
    {"device_id": "home-101", "timestamp": "2026-10-22T03:00", "temperature": 22.0, "humidity": 82, "occupancy": 4, "prev_kwh": 0.65},
    # Case 3: single resident, weekday 11:00, nobody home
    {"device_id": "home-103", "timestamp": "2026-10-21T11:00", "temperature": 29.5, "humidity": 66, "occupancy": 0, "prev_kwh": 0.16},
    # Case 4: AC-heavy couple, Saturday 15:00, very hot, both home
    {"device_id": "home-102", "timestamp": "2026-10-24T15:00", "temperature": 35.0, "humidity": 55, "occupancy": 2, "prev_kwh": 2.6},
]

X = pd.DataFrame([make_features(s) for s in samples], columns=FEATURES)
for i, (s, kwh) in enumerate(zip(samples, model.predict(X)), start=1):
    print(f"Case {i}: {s['device_id']} {s['timestamp']} T={s['temperature']}°C occ={s['occupancy']} "
          f"last hour={s['prev_kwh']} kWh -> next hour {kwh:.2f} kWh ({load_level(kwh)})")
