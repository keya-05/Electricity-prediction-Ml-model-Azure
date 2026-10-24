"""Simulated IoT weather stations.

Each station drifts between dry spells and storm fronts, so its readings (and the
model's predictions) change over time. Every tick, all stations send one reading.

Targets:
  http   (default) - POST each reading to the edge module
  iothub           - send each reading to Azure IoT Hub as a device-to-cloud message
                     (set IOTHUB_DEVICE_CONNECTION_STRING)

Usage:
  python weather_simulator.py
  python weather_simulator.py --interval 1 --count 100
  python weather_simulator.py --target iothub
"""
import argparse
import json
import random
import time
from datetime import datetime, timezone

import requests

from config import EDGE_URL, IOTHUB_DEVICE_CONNECTION_STRING

# Typical sensor values in dry weather and inside a rain front
DRY = {"temperature": 31.0, "humidity": 45.0, "pressure": 1014.0, "wind_speed": 9.0}
WET = {"temperature": 21.0, "humidity": 89.0, "pressure": 1001.0, "wind_speed": 25.0}
NOISE = {"temperature": 0.8, "humidity": 3.0, "pressure": 1.2, "wind_speed": 2.5}
LIMITS = {"temperature": (-10, 50), "humidity": (0, 100), "pressure": (950, 1050), "wind_speed": (0, 120)}

# Per-station offsets so the three sites behave differently
STATIONS = {
    "ws-pune":   {"temperature": 0.0, "humidity": -4.0, "pressure": 1.0, "wind_speed": 0.0},
    "ws-mumbai": {"temperature": 1.5, "humidity": 6.0, "pressure": -1.5, "wind_speed": 3.0},
    "ws-nashik": {"temperature": -1.0, "humidity": -8.0, "pressure": 2.0, "wind_speed": -2.0},
}


class Station:
    def __init__(self, device_id, offsets, rng):
        self.device_id = device_id
        self.offsets = offsets
        self.rng = rng
        self.storminess = rng.uniform(0.1, 0.9)  # 0 = clear sky, 1 = heavy rain front
        self.target = rng.choice([0.15, 0.85])

    def step(self):
        if self.rng.random() < 0.06:  # a front arrives or clears
            self.target = 0.85 if self.target < 0.5 else 0.15
        self.storminess += 0.15 * (self.target - self.storminess) + self.rng.gauss(0, 0.04)
        self.storminess = min(max(self.storminess, 0.0), 1.0)

    def reading(self, timestamp):
        s = self.storminess
        values = {}
        for key in DRY:
            value = DRY[key] + s * (WET[key] - DRY[key]) + self.offsets[key]
            value += self.rng.gauss(0, NOISE[key])
            lo, hi = LIMITS[key]
            values[key] = round(min(max(value, lo), hi), 1)
        return {"device_id": self.device_id, "timestamp": timestamp, **values}


class HttpSender:
    def __init__(self):
        self.session = requests.Session()

    def send(self, reading):
        r = self.session.post(f"{EDGE_URL}/telemetry", json=reading, timeout=5)
        r.raise_for_status()
        result = r.json()
        return f"{result['prediction']} ({result['rain_probability']:.0%})"

    def close(self):
        self.session.close()


class IoTHubSender:
    def __init__(self):
        if not IOTHUB_DEVICE_CONNECTION_STRING:
            raise SystemExit("Set IOTHUB_DEVICE_CONNECTION_STRING in .env to use --target iothub")
        from azure.iot.device import IoTHubDeviceClient

        self.client = IoTHubDeviceClient.create_from_connection_string(IOTHUB_DEVICE_CONNECTION_STRING)
        self.client.connect()

    def send(self, reading):
        from azure.iot.device import Message

        msg = Message(json.dumps(reading))
        msg.content_type = "application/json"
        msg.content_encoding = "utf-8"
        self.client.send_message(msg)
        return "sent to IoT Hub"

    def close(self):
        self.client.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Simulated IoT weather stations")
    parser.add_argument("--target", choices=["http", "iothub"], default="http")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between ticks")
    parser.add_argument("--count", type=int, default=0, help="number of ticks (0 = run forever)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    stations = [Station(device_id, offsets, rng) for device_id, offsets in STATIONS.items()]
    sender = IoTHubSender() if args.target == "iothub" else HttpSender()
    print(f"[sim] {len(stations)} stations -> {args.target}, every {args.interval}s (Ctrl+C to stop)")

    tick = 0
    try:
        while args.count == 0 or tick < args.count:
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for station in stations:
                station.step()
                reading = station.reading(timestamp)
                try:
                    result = sender.send(reading)
                except requests.RequestException as exc:
                    result = f"send failed: {exc.__class__.__name__} (is the edge module running?)"
                print(
                    f"[sim] {reading['device_id']:<10} T={reading['temperature']:>5.1f} "
                    f"H={reading['humidity']:>5.1f} P={reading['pressure']:>7.1f} "
                    f"W={reading['wind_speed']:>5.1f} -> {result}"
                )
            tick += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[sim] stopped")
    finally:
        sender.close()


if __name__ == "__main__":
    main()
