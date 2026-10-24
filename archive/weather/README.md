# Weather Edge AI – Real-time Rain Prediction for IoT/Edge

An IoT/Edge application (domain: **environment / weather monitoring**) that uses a Machine Learning model to predict **Rain / No Rain** in real time from weather-station sensor data.

```
 Weather Simulator ──► Edge Module (Decision Tree) ──► SQLite store ──► Live Dashboard
 3 IoT stations         validate + predict + store                       (Streamlit)
        │                        ▲
        └─► Azure IoT Hub ───────┘  (optional cloud path)
                                 ▲
             Azure ML: train job + Model Registry (weather-model)
```

## Project files

| File | Purpose |
|---|---|
| `weather_data.csv` | Dataset: 120 labelled readings (temperature, humidity, pressure, wind speed → Rain / No Rain) |
| `train_model.py` | Trains the Decision Tree, prints accuracy / confusion matrix, saves `weather_model.pkl` + `model_metrics.json` |
| `test_model.py` | Four sample predictions against the trained model |
| `weather_simulator.py` | Simulates 3 IoT weather stations (Pune, Mumbai, Nashik) sending readings every 2 s |
| `edge_module.py` | Edge inference service: receives readings, runs the model, stores predictions |
| `dashboard.py` | Live Streamlit dashboard |
| `storage.py`, `config.py` | Shared prediction store and settings |
| `azure_ml.py` | Azure ML: submit training job, register / download the model |
| `conda.yml` | Environment for the Azure ML training job |
| `Dockerfile`, `requirements-edge.txt` | Container image for deploying the edge module as an IoT Edge module |
| `run_demo.ps1` | Starts edge module + simulator + dashboard in one go |
| `docs/azure_ml_lab_guide.md` | Step-by-step Azure ML Studio guide |

## 1. Setup (once)

```powershell
cd D:\ML-model-AEAL
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation fails with "running scripts is disabled": `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 2. Train the model

```powershell
python train_model.py     # -> Model Accuracy: 0.75, saves weather_model.pkl
python test_model.py      # -> 4 sample predictions
```

## 3. Run the real-time system

**One command:** `.\run_demo.ps1` opens three windows and the dashboard at <http://localhost:8501>.

**Or manually**, in three terminals (activate the venv in each):

```powershell
python edge_module.py              # terminal 1 – edge inference on http://127.0.0.1:5000
python weather_simulator.py        # terminal 2 – IoT data stream
streamlit run dashboard.py         # terminal 3 – dashboard on http://localhost:8501
```

Simulator options: `--interval 1` (seconds between readings), `--count 100` (stop after 100 rounds), `--seed 42` (repeatable run).

Edge module API:

| Endpoint | Description |
|---|---|
| `POST /telemetry` | Body `{"device_id", "temperature", "humidity", "pressure", "wind_speed", "timestamp"}` → prediction |
| `GET /predictions?limit=50` | Latest stored predictions |
| `GET /health` | Liveness check (used by the dashboard's Online/Offline indicator) |

## 4. Azure (configure at the end)

```powershell
pip install -r requirements-azure.txt
copy .env.example .env      # then fill in the values
az login
```

**Azure ML – train and register the model**

```powershell
python azure_ml.py submit-job --wait               # trains on weather-ml-compute
python azure_ml.py register --from-job <job-name>  # registers weather-model:1
#   or register the local file:  python azure_ml.py register
python azure_ml.py download                        # edge module now uses the registry model
```

The same steps through the Studio UI are in [docs/azure_ml_lab_guide.md](docs/azure_ml_lab_guide.md).

**Azure IoT Hub – cloud data path**

1. Create an IoT Hub (Free tier F1) in an allowed region, add a device `weather-station-01`.
2. Put the device's connection string in `IOTHUB_DEVICE_CONNECTION_STRING` and the hub's
   *Built-in endpoints → Event Hub-compatible endpoint* in `IOTHUB_EVENTHUB_CONNECTION_STRING`.
3. Run:

```powershell
python edge_module.py --source iothub
python weather_simulator.py --target iothub
streamlit run dashboard.py
```

The prediction log's `source` column shows `iothub` for messages that came through the cloud.

**Optional – container for IoT Edge**

```powershell
docker build -t weather-edge-module .
docker run -p 5000:5000 weather-edge-module
```
