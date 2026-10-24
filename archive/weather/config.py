"""Shared settings for the simulator, edge module and dashboard.

Values can be overridden with environment variables or a .env file
(copy .env.example to .env).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

# Model
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "weather_model.pkl"))
FEATURES = ["temperature", "humidity", "pressure", "wind_speed"]
RAIN_LABEL = "Rain"

# Prediction store shared by the edge module and the dashboard
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "data" / "predictions.db"))

# Edge module HTTP endpoint (local mode)
EDGE_HOST = os.getenv("EDGE_HOST", "127.0.0.1")
EDGE_PORT = int(os.getenv("EDGE_PORT", "5000"))
EDGE_URL = os.getenv("EDGE_URL", f"http://{EDGE_HOST}:{EDGE_PORT}")

# Azure IoT Hub (optional - only needed for --target iothub / --source iothub)
IOTHUB_DEVICE_CONNECTION_STRING = os.getenv("IOTHUB_DEVICE_CONNECTION_STRING", "")
IOTHUB_EVENTHUB_CONNECTION_STRING = os.getenv("IOTHUB_EVENTHUB_CONNECTION_STRING", "")
IOTHUB_CONSUMER_GROUP = os.getenv("IOTHUB_CONSUMER_GROUP", "$Default")

# Azure Machine Learning (optional - only needed for the azure_*.py scripts)
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "")
AZURE_ML_WORKSPACE = os.getenv("AZURE_ML_WORKSPACE", "")
AZURE_ML_COMPUTE = os.getenv("AZURE_ML_COMPUTE", "weather-ml-compute")
AZURE_MODEL_NAME = os.getenv("AZURE_MODEL_NAME", "weather-model")
