
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible random dataset
rng = np.random.default_rng(42)

# Computational grid
x = np.linspace(-2, 2, 21)
y = np.linspace(-2, 2, 21)
X, Y = np.meshgrid(x, y)

# Dataset settings
SAMPLES_PER_CLASS = 300
MEASUREMENT_NOISE = 0.001

# Extract numerical features from a velocity field
def extract_features(u, v):

    du_dy, du_dx = np.gradient(u, y, x)
    dv_dy, dv_dx = np.gradient(v, y, x)

    # Two-dimensional vorticity
    omega = dv_dx - du_dy

    speed = np.sqrt(u**2 + v**2)

    return {
        "mean_vorticity": np.mean(omega),
        "mean_absolute_vorticity": np.mean(np.abs(omega)),
        "vorticity_std": np.std(omega),
        "maximum_absolute_vorticity": np.max(np.abs(omega)),
        "mean_velocity": np.mean(speed),
        "velocity_std": np.std(speed),
        "mean_du_dy": np.mean(np.abs(du_dy)),
        "mean_dv_dx": np.mean(np.abs(dv_dx))
    }


# Generate different velocity fields
def generate_flow(flow_type):

    U = rng.uniform(-2, 2)
    V = rng.uniform(-2, 2)

    if flow_type == "uniform":
        u = U + np.zeros_like(X)
        v = V + np.zeros_like(Y)

    elif flow_type == "linear_strain":
        a = rng.uniform(0.3, 1.5)
        u = a * X + U
        v = -a * Y + V

    elif flow_type == "source_sink":
        k = rng.uniform(0.3, 1.5)
        u = k * X + U
        v = k * Y + V

    elif flow_type == "simple_shear":
        a = rng.uniform(0.3, 2.0)
        u = a * Y + U
        v = np.zeros_like(Y) + V

    elif flow_type == "solid_body":
        omega = rng.choice([-1, 1]) * rng.uniform(0.3, 2.0)
        u = -omega * Y + U
        v = omega * X + V

    # Add small measurement noise
    u = u + rng.normal(0, MEASUREMENT_NOISE, u.shape)
    v = v + rng.normal(0, MEASUREMENT_NOISE, v.shape)

    return u, v


# Flow families and their correct labels
flow_types = {
    "uniform": 0,
    "linear_strain": 0,
    "source_sink": 0,
    "simple_shear": 1,
    "solid_body": 1
}

# Generate dataset
records = []

for flow_type, label in flow_types.items():

    for sample in range(SAMPLES_PER_CLASS):

        u, v = generate_flow(flow_type)

        features = extract_features(u, v)

        features["flow_type"] = flow_type
        features["label"] = label

        records.append(features)


# Save dataset
df = pd.DataFrame(records)

output_folder = Path("data")
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "flow_training_dataset.csv"
df.to_csv(output_file, index=False)

print("\nDATASET GENERATED SUCCESSFULLY!")
print("--------------------------------")
print("Total samples:", len(df))
print("\nFlow distribution:")
print(df["flow_type"].value_counts())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nDataset saved at:", output_file)
print("\nFirst five samples:")
print(df.head())
