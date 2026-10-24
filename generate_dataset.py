"""Generate energy_data.csv: 180 days of hourly smart-meter data for three homes.

Usage:
  python generate_dataset.py
"""
import csv
import random
from datetime import datetime

from features import FEATURES, make_features
from household import HOMES, Home

START = datetime(2026, 4, 1)
DAYS = 180
OUT = "energy_data.csv"


def main():
    rng = random.Random(42)
    rows = []
    for device_id in HOMES:
        home = Home(device_id, rng, START)
        home.next_hour()  # warm-up so the first row has a real prev_kwh
        for _ in range(DAYS * 24 - 1):
            reading, kwh = home.next_hour()
            rows.append({"device_id": device_id, "timestamp": reading["timestamp"],
                         **make_features(reading), "kwh": kwh})

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["device_id", "timestamp", *FEATURES, "kwh"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows ({len(HOMES)} homes x {DAYS} days, hourly) to {OUT}")


if __name__ == "__main__":
    main()
