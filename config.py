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
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "energy_model.pkl"))

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

# Azure Machine Learning (optional - only needed for the azure_ml.py script)
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "")
AZURE_ML_WORKSPACE = os.getenv("AZURE_ML_WORKSPACE", "")
AZURE_ML_COMPUTE = os.getenv("AZURE_ML_COMPUTE", "weather-ml-compute")
AZURE_MODEL_NAME = os.getenv("AZURE_MODEL_NAME", "energy-model")
# Endpoint names must be unique within the Azure region, so the default adds part of the subscription id
AZURE_ENDPOINT_NAME = os.getenv("AZURE_ENDPOINT_NAME", f"energy-ep-{AZURE_SUBSCRIPTION_ID[:8]}")
AZURE_ENDPOINT_INSTANCE_TYPE = os.getenv("AZURE_ENDPOINT_INSTANCE_TYPE", "Standard_DS2_v2")

# Deployed Azure ML endpoint (optional - only needed for edge_module.py --predictor endpoint)
AZURE_ENDPOINT_URL = os.getenv("AZURE_ENDPOINT_URL", "")
AZURE_ENDPOINT_KEY = os.getenv("AZURE_ENDPOINT_KEY", "")
