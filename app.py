
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="AI Flow Classification",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 AI-Based Rotational and Irrotational Flow Classification")
st.markdown(
    "An interactive fluid mechanics simulation "
    "and machine learning classification tool."
)

st.divider()

# ---------------------------------
# LOAD TRAINED MODEL
# ---------------------------------

MODEL_PATH = Path("models/flow_classifier.pkl")

if not MODEL_PATH.exists():
    st.error("Trained AI model not found!")
    st.stop()

model = joblib.load(MODEL_PATH)

# ---------------------------------
# FEATURE EXTRACTION
# ---------------------------------

FEATURES = [
    "mean_vorticity",
    "mean_absolute_vorticity",
    "vorticity_std",
    "maximum_absolute_vorticity",
    "mean_velocity",
    "velocity_std",
    "mean_du_dy",
    "mean_dv_dx"
]

def extract_features(u, v, x, y):

    du_dy, du_dx = np.gradient(u, y, x)
    dv_dy, dv_dx = np.gradient(v, y, x)

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
    }, omega

# ---------------------------------
# SIDEBAR PARAMETERS
# ---------------------------------

st.sidebar.header("Simulation Controls")

flow_type = st.sidebar.selectbox(
    "Select Flow Type",
    [
        "Uniform Flow",
        "Linear Strain",
        "Source / Sink",
        "Simple Shear",
        "Solid-Body Rotation",
        "Lamb-Oseen Vortex"
    ]
)

st.sidebar.subheader("Flow Parameters")

U = st.sidebar.slider("Horizontal velocity U", -2.0, 2.0, 1.0)
V = st.sidebar.slider("Vertical velocity V", -2.0, 2.0, 0.0)

strength = st.sidebar.slider(
    "Flow strength",
    0.1, 2.0, 1.0
)

Gamma = st.sidebar.slider(
    "Vortex circulation",
    1.0, 10.0, 5.0
)

core_radius = st.sidebar.slider(
    "Vortex core radius",
    0.3, 1.5, 0.8
)

# ---------------------------------
# COMPUTATIONAL GRID
# ---------------------------------

x = np.linspace(-2, 2, 41)
y = np.linspace(-2, 2, 41)

X, Y = np.meshgrid(x, y)

# ---------------------------------
# GENERATE VELOCITY FIELDS
# ---------------------------------

if flow_type == "Uniform Flow":

    u = U + np.zeros_like(X)
    v = V + np.zeros_like(Y)

elif flow_type == "Linear Strain":

    u = strength * X + U
    v = -strength * Y + V

elif flow_type == "Source / Sink":

    u = strength * X + U
    v = strength * Y + V

elif flow_type == "Simple Shear":

    u = strength * Y + U
    v = np.zeros_like(Y) + V

elif flow_type == "Solid-Body Rotation":

    u = -strength * Y + U
    v = strength * X + V

elif flow_type == "Lamb-Oseen Vortex":

    r2 = X**2 + Y**2

    factor = np.empty_like(r2)
    mask = r2 > 1e-12

    factor[mask] = (
        Gamma / (2 * np.pi * r2[mask])
        * (1 - np.exp(-r2[mask] / core_radius**2))
    )

    factor[~mask] = (
        Gamma / (2 * np.pi * core_radius**2)
    )

    u = -factor * Y
    v = factor * X

# ---------------------------------
# CALCULATE FLUID MECHANICS FEATURES
# ---------------------------------

features, omega = extract_features(u, v, x, y)

input_data = pd.DataFrame(
    [features],
    columns=FEATURES
)

# ---------------------------------
# AI PREDICTION
# ---------------------------------

prediction = model.predict(input_data)[0]

probabilities = model.predict_proba(input_data)[0]

confidence = probabilities[
    np.where(model.classes_ == prediction)[0][0]
]

labels = {
    0: "IRROTATIONAL FLOW",
    1: "ROTATIONAL FLOW"
}

# ---------------------------------
# DISPLAY RESULTS
# ---------------------------------

st.subheader("AI Classification Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Selected Flow", flow_type)

with col2:
    st.metric("AI Prediction", labels[prediction])

with col3:
    st.metric(
        "Prediction Confidence",
        f"{confidence * 100:.2f}%"
    )

st.progress(float(confidence))

# ---------------------------------
# VISUALIZATION
# ---------------------------------

st.divider()
st.subheader("Fluid Flow Visualization")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

speed = np.sqrt(u**2 + v**2)

# Velocity streamlines
axes[0].streamplot(
    X, Y, u, v,
    color=speed,
    cmap="viridis",
    density=1.4
)

axes[0].set_title("Velocity Streamlines")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].set_aspect("equal")

# Vorticity contour
contour = axes[1].contourf(
    X, Y, omega,
    levels=30,
    cmap="coolwarm"
)

fig.colorbar(
    contour,
    ax=axes[1],
    label="Vorticity (1/s)"
)

axes[1].set_title("Vorticity Distribution")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
axes[1].set_aspect("equal")

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# ---------------------------------
# NUMERICAL RESULTS
# ---------------------------------

st.divider()
st.subheader("Fluid Mechanics Parameters")

result_col1, result_col2 = st.columns(2)

with result_col1:
    st.metric(
        "Mean Absolute Vorticity",
        f"{features['mean_absolute_vorticity']:.5f}"
    )

    st.metric(
        "Maximum Absolute Vorticity",
        f"{features['maximum_absolute_vorticity']:.5f}"
    )

with result_col2:
    st.metric(
        "Mean Velocity",
        f"{features['mean_velocity']:.5f}"
    )

    st.metric(
        "Vorticity Standard Deviation",
        f"{features['vorticity_std']:.5f}"
    )

st.info(
    "Physical interpretation: In a two-dimensional flow, "
    "zero vorticity everywhere indicates irrotational flow. "
    "Non-zero vorticity indicates rotational flow."
)

st.caption(
    "AI predictions depend on the training data and are not "
    "a substitute for physical validation."
)
