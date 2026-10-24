# Smart Energy Edge AI – Electricity Consumption Prediction on Azure + IoT/Edge

A Machine Learning model, trained and deployed with **Azure Machine Learning**, predicts how much electricity (kWh) each home will use in the **coming hour**. Simulated IoT smart meters stream readings; an edge module (or the Azure endpoint) predicts in real time; a live dashboard shows forecasts, actual vs predicted use and unusual-usage alerts.

```
 Smart-meter simulator ──► Edge module ──────────────► SQLite ──► Live dashboard
 3 homes, 1 h every 2 s     predictor: local model          (Streamlit)
        │                     or Azure ML endpoint ◄─┐
        └─► Azure IoT Hub ─────────┘ (optional)      │ REST (key auth)
                                                      │
        Azure ML: training job → Model Registry → Managed online endpoint
```

## Project files

| File | Purpose |
|---|---|
| `household.py` | Simulated homes: residents, routine, lighting, air-conditioning |
| `generate_dataset.py` → `energy_data.csv` | 12,957 hourly readings (3 homes × 180 days) |
| `features.py` | Feature engineering shared by training, edge module and Azure endpoint |
| `train_model.py` → `energy_model.pkl`, `model_metrics.json` | Random Forest Regressor, time-based split, MAE / RMSE / R² vs naive baseline |
| `test_model.py` | Four sample predictions |
| `energy_simulator.py` | IoT smart meters (with occasional "appliance left on" events) |
| `edge_module.py` | Edge inference service (`--predictor local` or `endpoint`, `--source http` or `iothub`) |
| `dashboard.py` | Live Streamlit dashboard |
| `azure_ml.py` | Azure ML: training job, registry, online endpoint deploy / test / delete |
| `azure_deploy/` | Endpoint scoring script, inference environment, sample request |
| `conda.yml` | Environment for the Azure ML training job |
| `Dockerfile`, `iotedge/deployment.template.json` | Container + manifest for deploying the edge module to an IoT Edge device |
| `run_demo.ps1` | Starts edge module + simulator + dashboard |
| `archive/weather/` | Previous version of the project (rain prediction) |

## 1. Run locally

```powershell
cd D:\ML-model-AEAL
.\.venv\Scripts\Activate.ps1          # first time: python -m venv .venv; pip install -r requirements.txt
python generate_dataset.py            # -> energy_data.csv
python train_model.py                 # -> MAE 0.074 kWh, R2 0.98, saves energy_model.pkl
python test_model.py                  # -> 4 sample forecasts
.\run_demo.ps1                        # edge module + simulator + dashboard at http://localhost:8501
```

## 2. Deploy to Azure

Prerequisites (already done for this project): workspace `aml-weather-lab` in **Central India**, resource group `IoTEdgeLabRG`, compute instance `weather-ml-compute`, `.env` filled in, `pip install -r requirements-azure.txt`, `az login`.

| Step | Command | What happens |
|---|---|---|
| 1. Start compute | Studio → Compute → `weather-ml-compute` → Start | VM that runs the training job |
| 2. Train in Azure | `python azure_ml.py submit-job --wait` | Uploads `train_model.py` + `energy_data.csv`, builds the conda environment, trains, saves `outputs/energy_model.pkl` (~5 min) |
| 3. Register | `python azure_ml.py register --from-job <job-name>` | Creates `energy-model:1` in the Model Registry |
| 4. Stop compute | Studio → Compute → Stop | Not needed for deployment |
| 5. Deploy | `python azure_ml.py deploy-endpoint` | Creates a managed online endpoint and deployment `blue` (1 × Standard_DS2_v2) running `score.py` (10–20 min) |
| 6. Test | `python azure_ml.py test-endpoint` | Sends `azure_deploy/sample_request.json`, prints the forecasts |
| 7. Connect | `python azure_ml.py endpoint-info --save` | Writes the scoring URL and key to `.env` |
| 8. Use it live | `.\run_demo.ps1 -Endpoint` | Edge module calls the Azure endpoint for every reading; dashboard shows Predictor = Endpoint |
| 9. Clean up | `python azure_ml.py delete-endpoint` | **The endpoint is billed every hour it exists** (~USD 0.15–0.20/h) – delete it after the demo |

Troubleshooting: `python azure_ml.py endpoint-logs` shows the container log if step 5 fails. A quota error means the region has too few vCPUs for the instance type – try `--instance-type Standard_DS1_v2` or stop the compute instance first. An endpoint-name conflict → set `AZURE_ENDPOINT_NAME=energy-ep-<yourname>` in `.env`.

Endpoint request / response:

```json
POST <scoring-url>   Authorization: Bearer <key>
{"data": [{"device_id": "home-101", "timestamp": "2026-10-21T19:00", "temperature": 31.0,
           "humidity": 62, "occupancy": 4, "prev_kwh": 2.4}]}

{"predictions": [3.338], "load_levels": ["Peak"]}
```

## 3. Optional: Azure IoT Hub (cloud data path)

1. Create an IoT Hub (Free F1) in Central India and a device `smart-meter-01`.
2. In `.env`: `IOTHUB_DEVICE_CONNECTION_STRING` (device → primary connection string) and `IOTHUB_EVENTHUB_CONNECTION_STRING` (hub → Built-in endpoints → Event Hub-compatible endpoint).
3. `python edge_module.py --source iothub`, `python energy_simulator.py --target iothub`, `streamlit run dashboard.py`.

## 4. Optional: Azure IoT Edge device

Needs an Azure Container Registry and a Linux device/VM with the IoT Edge runtime.

```powershell
az acr create -g IoTEdgeLabRG -n <acrname> --sku Basic --admin-enabled true -l centralindia
az acr build -r <acrname> -t energy-edge-module:1.0 .
az iot hub device-identity create --hub-name <hub> --device-id energy-edge-device --edge-enabled
# fill <ACR_NAME>, <ACR_USERNAME>, <ACR_PASSWORD> in iotedge/deployment.template.json, then:
az iot edge set-modules --hub-name <hub> --device-id energy-edge-device --content iotedge/deployment.template.json
```
