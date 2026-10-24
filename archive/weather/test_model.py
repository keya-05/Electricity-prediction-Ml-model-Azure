import joblib
import pandas as pd

# Load the trained model
model = joblib.load("weather_model.pkl")

# Sample sensor readings: temperature (°C), humidity (%), pressure (hPa), wind_speed (km/h)
samples = pd.DataFrame(
    [
        [21.5, 90, 1000.2, 28.0],  # Case 1: humid, low pressure, windy
        [33.0, 40, 1015.8, 8.0],   # Case 2: hot, dry, high pressure, calm
        [24.0, 85, 1003.5, 18.0],  # Case 3: mild, humid, falling pressure
        [27.5, 55, 1012.0, 12.0],  # Case 4: warm, moderate humidity
    ],
    columns=["temperature", "humidity", "pressure", "wind_speed"],
)

for i, (row, pred) in enumerate(zip(samples.values.tolist(), model.predict(samples)), start=1):
    print(f"Case {i}: {row} -> {pred}")
