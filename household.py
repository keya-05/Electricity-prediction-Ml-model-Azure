"""Simulated smart homes, shared by the dataset generator and the IoT simulator.

Each home has residents, a base load (fridge, router, standby devices), appliance use
that follows the daily routine (breakfast, evening cooking/TV), lighting after dark,
and air-conditioning that runs when it is hot and someone is home.
"""
import math
from datetime import timedelta

HOMES = {
    "home-101": {"label": "Family of 4", "residents": 4, "base": 0.35, "appliance": 0.55, "ac_kw": 1.6},
    "home-102": {"label": "Couple, AC-heavy", "residents": 2, "base": 0.25, "appliance": 0.40, "ac_kw": 2.2},
    "home-103": {"label": "Single resident", "residents": 1, "base": 0.15, "appliance": 0.30, "ac_kw": 1.0},
}


class Weather:
    """Outdoor temperature (°C) and humidity (%) with a daily cycle, a seasonal trend and day-to-day variation."""

    def __init__(self, rng):
        self.rng = rng
        self.offset = rng.gauss(0, 1.5)

    def at(self, ts):
        if ts.hour == 0:
            self.offset = 0.7 * self.offset + self.rng.gauss(0, 1.2)
        seasonal = 3 * math.sin(2 * math.pi * (ts.timetuple().tm_yday - 100) / 365)
        daily = 5.5 * math.cos(2 * math.pi * (ts.hour - 15) / 24)  # warmest at 15:00
        temperature = 27 + seasonal + daily + self.offset + self.rng.gauss(0, 0.6)
        humidity = 70 - 1.8 * (temperature - 27) + self.rng.gauss(0, 4)
        return round(temperature, 1), round(min(max(humidity, 20), 100), 1)


def presence_probability(hour, weekend, residents):
    if hour < 7 or hour >= 22:
        return 0.97
    if weekend:
        return 0.75 if 10 <= hour < 18 else 0.9
    if 9 <= hour < 17:
        return 0.2 if residents > 1 else 0.05  # work / school hours
    return 0.85


def hourly_consumption(home, hour, temperature, occupancy, rng):
    awake = 6 <= hour < 23
    share = occupancy / home["residents"]
    kwh = home["base"]
    if awake:
        kwh += 0.10 * occupancy
    if 6 <= hour < 9:
        kwh += home["appliance"] * 1.0 * share  # breakfast, water heater
    elif 18 <= hour < 22:
        kwh += home["appliance"] * 1.6 * share  # cooking, TV, washing machine
    elif 9 <= hour < 18:
        kwh += home["appliance"] * 0.4 * share
    if (hour >= 18 or hour < 6) and occupancy:
        kwh += 0.08 * occupancy  # lighting
    threshold = 26 if awake else 27
    if occupancy and temperature > threshold:
        kwh += home["ac_kw"] * min(1.0, (temperature - threshold) / 6)
    kwh *= 1 + rng.gauss(0, 0.06)
    return max(kwh, 0.05)


class Home:
    """Steps one home forward hour by hour."""

    def __init__(self, device_id, rng, start):
        self.device_id = device_id
        self.profile = HOMES[device_id]
        self.rng = rng
        self.weather = Weather(rng)
        self.ts = start
        self.prev_kwh = None

    def next_hour(self, extra_kwh=0.0):
        """Return (reading at the start of the hour, actual kWh used during that hour)."""
        ts = self.ts
        temperature, humidity = self.weather.at(ts)
        residents = self.profile["residents"]
        p = presence_probability(ts.hour, ts.weekday() >= 5, residents)
        occupancy = sum(self.rng.random() < p for _ in range(residents))
        kwh = round(hourly_consumption(self.profile, ts.hour, temperature, occupancy, self.rng) + extra_kwh, 3)
        reading = {
            "device_id": self.device_id,
            "timestamp": ts.isoformat(timespec="minutes"),
            "temperature": temperature,
            "humidity": humidity,
            "occupancy": occupancy,
            "prev_kwh": self.prev_kwh if self.prev_kwh is not None else kwh,
        }
        self.prev_kwh = kwh
        self.ts = ts + timedelta(hours=1)
        return reading, kwh
