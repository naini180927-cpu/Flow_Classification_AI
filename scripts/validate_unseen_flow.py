
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

# Load trained AI model
model = joblib.load("models/flow_classifier.pkl")

# Create computational grid
x = np.linspace(-2, 2, 41)
y = np.linspace(-2, 2, 41)
X, Y = np.meshgrid(x, y)

# Lamb-Oseen vortex parameters
Gamma = 5.0
core_radius = 0.8

r2 = X**2 + Y**2

# Avoid division by zero at the vortex centre
factor = np.empty_like(r2)
mask = r2 > 1e-12

factor[mask] = (
    Gamma / (2 * np.pi * r2[mask])
    * (1 - np.exp(-r2[mask] / core_radius**2))
)

factor[~mask] = Gamma / (2 * np.pi * core_radius**2)

# Velocity components
u = -factor * Y
v = factor * X

# Calculate velocity gradients
du_dy, du_dx = np.gradient(u, y, x)
dv_dy, dv_dx = np.gradient(v, y, x)

# Calculate vorticity
omega = dv_dx - du_dy

# Extract the same features used during training
features = pd.DataFrame([{
    "mean_vorticity": np.mean(omega),
    "mean_absolute_vorticity": np.mean(np.abs(omega)),
    "vorticity_std": np.std(omega),
    "maximum_absolute_vorticity": np.max(np.abs(omega)),
    "mean_velocity": np.mean(np.sqrt(u**2 + v**2)),
    "velocity_std": np.std(np.sqrt(u**2 + v**2)),
    "mean_du_dy": np.mean(np.abs(du_dy)),
    "mean_dv_dx": np.mean(np.abs(dv_dx))
}])

# AI prediction
prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]

classes = model.classes_
confidence = probabilities[np.where(classes == prediction)[0][0]]

labels = {
    0: "IRROTATIONAL",
    1: "ROTATIONAL"
}

print("\nUNSEEN FLOW VALIDATION")
print("------------------------------")
print("Flow type: Lamb-Oseen Vortex")
print("AI prediction:", labels[prediction])
print("Prediction confidence:", round(confidence * 100, 2), "%")
print("Mean absolute vorticity:", np.mean(np.abs(omega)))

# Save visualization
Path("outputs").mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].streamplot(X, Y, u, v, density=1.3)
axes[0].set_title("Lamb-Oseen Vortex - Streamlines")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].set_aspect("equal")

contour = axes[1].contourf(
    X, Y, omega, levels=30, cmap="coolwarm"
)
axes[1].set_title("Lamb-Oseen Vortex - Vorticity")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
axes[1].set_aspect("equal")

fig.colorbar(contour, ax=axes[1], label="Vorticity (1/s)")

plt.suptitle("Unseen Flow AI Validation")
plt.tight_layout()

plt.savefig(
    "outputs/unseen_flow_validation.png",
    dpi=300
)

plt.show()

print("\nValidation visualization saved successfully!")
