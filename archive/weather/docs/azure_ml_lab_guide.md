# Experiment 07 – Building an Azure Machine Learning Model for an IoT Edge Module (Weather Data)

**Objective:** Learn how to collect weather sensor data, prepare a dataset, train a Machine Learning model in Azure ML, register the trained model, and prepare it for deployment to an IoT Edge module.

## 1. Overview & Architecture Workflow

In this lab, we will train a **Decision Tree Classifier** on weather station data (Temperature, Humidity, Pressure and Wind Speed) to predict the weather condition (**Rain** vs. **No Rain**).

```
   [ Weather Station Sensor Data ]
                │ (Temperature, Humidity, Pressure, Wind Speed)
                ▼
       [ Azure ML Dataset ]
                │
                ▼
     [ Azure ML Compute Job ]
                │ (Runs train_model.py)
                ▼
    [ Decision Tree Classifier ]
                │
                ▼
    [ Trained Model (.pkl) ]
                │
                ▼
   [ Azure ML Model Registry ]
                │
                ▼
      ( Next Step: IoT Edge Deployment )
```

## Project Files

| File | Purpose |
|---|---|
| `weather_data.csv` | Training dataset (120 weather readings) |
| `train_model.py` | Trains and evaluates the Decision Tree, saves `weather_model.pkl` |
| `conda.yml` | Runtime environment for the Azure ML job |
| `test_model.py` | Runs sample predictions against the trained model |

## Step 1 — Open Azure Machine Learning Studio

1. Navigate to [Azure Machine Learning Studio](https://ml.azure.com).
2. Sign in using your Azure credentials.
3. Select your designated workspace.

## Step 2 — Create an Azure ML Workspace

If a workspace does not already exist, create one via the Azure Portal:

1. Search for **Machine Learning** in the search bar.
2. Click **Create > Machine Learning**.
3. Configure the following properties:

| Setting | Example Value |
|---|---|
| Subscription | Your Azure Subscription |
| Resource Group | IOTEdgelabRG |
| Workspace Name | aml-weather-lab |
| Region | Central India |

4. Click **Review + Create → Create**. Once created, launch Azure ML Studio.

## Step 3 — Create the Training Dataset

The dataset is provided as `weather_data.csv` (120 rows). First rows:

```csv
temperature,humidity,pressure,wind_speed,weather
17.7,92,996.6,30.5,Rain
36.6,38,1006.6,9.8,Rain
20.7,98,1003.4,33.7,Rain
23.0,62,1012.3,19.6,Rain
26.6,34,1014.7,12.3,No Rain
```

**Dataset Structure**

| Column | Unit | Role |
|---|---|---|
| `temperature` | °C | Input feature (weather telemetry variable 1) |
| `humidity` | % | Input feature (weather telemetry variable 2) |
| `pressure` | hPa | Input feature (weather telemetry variable 3) |
| `wind_speed` | km/h | Input feature (weather telemetry variable 4) |
| `weather` | — | Target label (`Rain` or `No Rain`) |

Rainy readings typically show high humidity (> 72 %), low pressure (< 1009 hPa) and stronger winds; dry readings show the opposite. The ranges overlap slightly and about 8 % of readings are mislabelled (like rows 2 and 4 above, where a local shower occurred despite dry readings), so the model will not be 100 % accurate, just as with real sensors.

## Step 4 — Create a Compute Instance

1. In Azure ML Studio, click **Compute** from the left side menu.
2. Under **Compute instances**, click **New**.
3. Choose a low-cost virtual machine size (e.g., `Standard_DS11_v2`).
4. Set the compute name to: `weather-ml-compute`.
5. Click **Create** and wait until the state updates to **Running**.

## Step 5 — Upload Data Asset to Azure ML

1. Go to **Data → Create → Data asset**.
2. Select **From local files**.
3. Upload `weather_data.csv`.
4. Name the asset: `weather-sensor-data`.
5. Set the type to **Tabular** and complete the wizard.

## Step 6 — Create the Python Training Script (`train_model.py`)

```python
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv("weather_data.csv")

# 2. Define input features (X) and target label (y)
X = data[["temperature", "humidity", "pressure", "wind_speed"]]
y = data["weather"]

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 5. Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# 6. Export the serialized model file
joblib.dump(model, "weather_model.pkl")
print("Model saved successfully as weather_model.pkl")
```

`max_depth=4` keeps the tree small (fast on edge hardware) and limits overfitting; `stratify=y` keeps the Rain / No Rain ratio the same in the train and test sets.

## Step 7 — Define the Environment (`conda.yml`)

```yaml
name: weather-ml-environment
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      # same versions as the edge module, so the pickled model loads cleanly there
      - pandas==3.1.0
      - numpy==2.5.4
      - scikit-learn==1.9.1
      - joblib==1.6.0
```

## Step 8 — Submit and Run the Training Job

1. In Azure ML Studio, go to **Jobs → Create job**.
2. Select **ML Job**.
3. Set the execution command: `python train_model.py`
4. Attach `train_model.py` and `weather_data.csv`.
5. Target Compute: `weather-ml-compute`.
6. Set the Environment using `conda.yml`.
7. Click **Submit**.

## Step 9 — Evaluate Model Execution & Accuracy

1. Open the completed Job in Azure ML Studio.
2. Select **Outputs + logs** and review the execution logs (`std_log.txt`).
3. Verify output:

```
Model Accuracy: 0.75
              precision    recall  f1-score   support

     No Rain       0.73      0.85      0.79        13
        Rain       0.78      0.64      0.70        11

    accuracy                           0.75        24
   macro avg       0.76      0.74      0.74        24
weighted avg       0.75      0.75      0.75        24

Confusion matrix (rows = actual, cols = predicted, order = ['No Rain', 'Rain']):
[[11  2]
 [ 4  7]]
Feature importance: {'temperature': 0.154, 'humidity': 0.783, 'pressure': 0.062, 'wind_speed': 0.0}
Model saved successfully as weather_model.pkl
```

## Step 10 — Register Model in Registry

1. Navigate to **Models → Register → From local files**.
2. Upload `weather_model.pkl`.
3. Set Name: `weather-model`.
4. Set Version: `1`.
5. Click **Register**.

## Step 11 — Verify the Registered Model

Go to **Models → weather-model** and confirm a version such as `1` is listed.

This is the trained ML model that we will use in the next practical (IoT Edge deployment).

## Step 12 — Model Inference Logic (Sample Tests)

Run `python test_model.py` (with `weather_model.pkl` in the same folder) to check predictions:

| Scenario | Input (Temp °C, Humidity %, Pressure hPa, Wind km/h) | Expected Model Prediction |
|---|---|---|
| Case 1 | 21.5, 90, 1000.2, 28.0 | Rain |
| Case 2 | 33.0, 40, 1015.8, 8.0 | No Rain |
| Case 3 | 24.0, 85, 1003.5, 18.0 | Rain |
| Case 4 | 27.5, 55, 1012.0, 12.0 | No Rain |

## Summary Checklist

- [ ] Created Azure ML Workspace & Compute Instance
- [ ] Prepared Weather Telemetry CSV Data Asset
- [ ] Written `train_model.py` and environment definition (`conda.yml`)
- [ ] Executed Azure ML Job and verified accuracy
- [ ] Registered `weather-model:1` for IoT Edge container deployment
