import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["hour", "day_of_week", "is_weekend", "temperature", "humidity", "occupancy", "prev_kwh"]

# 1. Load dataset
data = pd.read_csv("energy_data.csv", parse_dates=["timestamp"]).sort_values("timestamp")

# 2. Time-based train/test split: train on the earlier 80% of the period, test on the latest 20%
cutoff = data["timestamp"].quantile(0.8)
train, test = data[data["timestamp"] < cutoff], data[data["timestamp"] >= cutoff]
X_train, y_train = train[FEATURES], train["kwh"]
X_test, y_test = test[FEATURES], test["kwh"]

# 3. Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
model.set_params(n_jobs=1)  # the edge predicts one reading at a time; parallel threads only add overhead

# 4. Model evaluation, against a naive baseline that predicts "same as the previous hour"
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)
baseline_mae = mean_absolute_error(y_test, test["prev_kwh"])
importance = dict(zip(FEATURES, model.feature_importances_.round(3).tolist()))
per_home = {
    device: round(mean_absolute_error(g["kwh"], model.predict(g[FEATURES])), 3)
    for device, g in test.groupby("device_id")
}

print(f"Train rows: {len(train)}, test rows: {len(test)} (test period from {cutoff:%Y-%m-%d})")
print(f"MAE:  {mae:.3f} kWh")
print(f"RMSE: {rmse:.3f} kWh")
print(f"R2:   {r2:.3f}")
print(f"Baseline MAE (same as previous hour): {baseline_mae:.3f} kWh "
      f"-> model is {100 * (1 - mae / baseline_mae):.0f}% more accurate")
print("MAE per home:", per_home)
print("Feature importance:", importance)

# 5. Export the serialized model file and its metrics
joblib.dump(model, "energy_model.pkl", compress=3)
metrics = {
    "model": "RandomForestRegressor(n_estimators=100, max_depth=12)",
    "mae": round(mae, 4),
    "rmse": round(rmse, 4),
    "r2": round(r2, 4),
    "baseline_mae": round(baseline_mae, 4),
    "train_rows": len(train),
    "test_rows": len(test),
    "test_from": f"{cutoff:%Y-%m-%d}",
    "mae_per_home": per_home,
    "feature_importance": importance,
}
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Azure ML keeps files written to ./outputs as job outputs
if "AZUREML_RUN_ID" in os.environ:
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/energy_model.pkl", compress=3)
    with open("outputs/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

print("Model saved successfully as energy_model.pkl")
