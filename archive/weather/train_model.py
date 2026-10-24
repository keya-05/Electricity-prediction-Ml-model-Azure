import json
import os

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES = ["temperature", "humidity", "pressure", "wind_speed"]

# 1. Load dataset
data = pd.read_csv("weather_data.csv")

# 2. Define input features (X) and target label (y)
X = data[FEATURES]
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
labels = list(model.classes_)
cm = confusion_matrix(y_test, y_pred, labels=labels)
importance = dict(zip(FEATURES, model.feature_importances_.round(3).tolist()))

print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
print(f"Confusion matrix (rows = actual, cols = predicted, order = {labels}):")
print(cm)
print("Feature importance:", importance)

# 6. Export the serialized model file and its metrics
joblib.dump(model, "weather_model.pkl")
metrics = {
    "model": "DecisionTreeClassifier(max_depth=4)",
    "accuracy": round(accuracy, 4),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "labels": labels,
    "confusion_matrix": cm.tolist(),
    "feature_importance": importance,
}
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Azure ML keeps files written to ./outputs as job outputs
if "AZUREML_RUN_ID" in os.environ:
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/weather_model.pkl")
    with open("outputs/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

print("Model saved successfully as weather_model.pkl")
