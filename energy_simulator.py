"""Simulated IoT smart meters for three homes.

Each tick is one simulated hour: every home sends its meter reading (energy used in the
hour that just ended) plus room sensors (temperature, humidity, occupancy). With the
default 2-second interval a full day passes in 48 seconds, so the daily load curve is
visible during a demo. Occasionally an appliance is "left on", adding unexpected load
that the dashboard flags as unusual usage.

Targets:
  http   (default) - POST each reading to the edge module
  iothub           - send each reading to Azure IoT Hub (set IOTHUB_DEVICE_CONNECTION_STRING)

Usage:
  python energy_simulator.py
  python energy_simulator.py --interval 1 --count 100
  python energy_simulator.py --target iothub
"""
import argparse
import json
import random
import time
from datetime import datetime

import requests

from config import EDGE_URL, IOTHUB_DEVICE_CONNECTION_STRING
from household import HOMES, Home

LEFT_ON_PROBABILITY = 0.015  # per home per hour


class HttpSender:
    def __init__(self):
        self.session = requests.Session()

    def send(self, reading):
        r = self.session.post(f"{EDGE_URL}/telemetry", json=reading, timeout=15)
        r.raise_for_status()
        result = r.json()
        return f"next hour {result['predicted_kwh']:.2f} kWh ({result['load_level']})"

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
    parser = argparse.ArgumentParser(description="Simulated IoT smart meters")
    parser.add_argument("--target", choices=["http", "iothub"], default="http")
    parser.add_argument("--interval", type=float, default=2.0, help="real seconds per simulated hour")
    parser.add_argument("--count", type=int, default=0, help="number of simulated hours (0 = run forever)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    start = datetime.now().replace(minute=0, second=0, microsecond=0)
    homes = [Home(device_id, rng, start) for device_id in HOMES]
    for home in homes:
        home.next_hour()  # warm-up so the first reading carries a real prev_kwh
    left_on = {home.device_id: (0, 0.0) for home in homes}  # hours remaining, extra kWh

    sender = IoTHubSender() if args.target == "iothub" else HttpSender()
    print(f"[sim] {len(homes)} smart meters -> {args.target}, 1 simulated hour every {args.interval}s (Ctrl+C to stop)")

    tick = 0
    try:
        while args.count == 0 or tick < args.count:
            for home in homes:
                hours, extra = left_on[home.device_id]
                if hours == 0 and rng.random() < LEFT_ON_PROBABILITY:
                    hours, extra = rng.randint(2, 4), round(rng.uniform(1.5, 2.5), 2)
                    print(f"[sim] {home.device_id}: appliance left on (+{extra} kWh for {hours} h)")
                reading, _ = home.next_hour(extra_kwh=extra if hours else 0.0)
                left_on[home.device_id] = (max(hours - 1, 0), extra)
                try:
                    result = sender.send(reading)
                except requests.RequestException as exc:
                    result = f"send failed: {exc.__class__.__name__} (is the edge module running?)"
                print(
                    f"[sim] {reading['device_id']:<9} {reading['timestamp']}  T={reading['temperature']:>4.1f} "
                    f"occ={reading['occupancy']}  last hour={reading['prev_kwh']:.2f} kWh -> {result}"
                )
            tick += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[sim] stopped")
    finally:
        sender.close()


if __name__ == "__main__":
    main()
