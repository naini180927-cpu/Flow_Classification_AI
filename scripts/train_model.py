
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load dataset
df = pd.read_csv("data/flow_training_dataset.csv")

# Select numerical input features
# Do not include flow_type or label as model inputs
features = [
    "mean_vorticity",
    "mean_absolute_vorticity",
    "vorticity_std",
    "maximum_absolute_vorticity",
    "mean_velocity",
    "velocity_std",
    "mean_du_dy",
    "mean_dv_dx"
]

X = df[features]
y = df["label"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train the model
print("\nTraining AI model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nAI MODEL TRAINING COMPLETED")
print("--------------------------------")
print("Model accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Irrotational", "Rotational"]
    )
)

# Create output folders
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# Save trained model
joblib.dump(model, "models/flow_classifier.pkl")

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Irrotational", "Rotational"]
)

display.plot(cmap="Blues")
plt.title("AI Flow Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()

# Feature importance graph
importance = model.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(features, importance)
plt.xlabel("Feature Importance")
plt.title("Importance of Fluid Mechanics Features")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=300)
plt.show()

print("\nSaved files:")
print("models/flow_classifier.pkl")
print("outputs/confusion_matrix.png")
print("outputs/feature_importance.png")
