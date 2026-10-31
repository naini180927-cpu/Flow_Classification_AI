
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Computational grid
x = np.linspace(-2, 2, 40)
y = np.linspace(-2, 2, 40)
X, Y = np.meshgrid(x, y)

# Flow parameters
U = 2.0
Omega = 1.0

# Uniform irrotational flow
u1 = U * np.ones_like(X)
v1 = np.zeros_like(Y)

# Solid-body rotational flow
u2 = -Omega * Y
v2 = Omega * X

# Vorticity calculation
def calculate_vorticity(u, v):
    dv_dy, dv_dx = np.gradient(v, y, x)
    du_dy, du_dx = np.gradient(u, y, x)
    return dv_dx - du_dy

w1 = calculate_vorticity(u1, v1)
w2 = calculate_vorticity(u2, v2)

# Remove negligible numerical noise for display
w1_plot = np.where(np.abs(w1) < 1e-10, 0, w1)
w2_plot = np.where(np.abs(w2 - 2*Omega) < 1e-10, 2*Omega, w2)

# Create four visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Uniform flow streamlines
axes[0, 0].streamplot(X, Y, u1, v1, density=1.2)
axes[0, 0].set_title("Uniform Irrotational Flow")

# Rotational flow streamlines
axes[0, 1].streamplot(X, Y, u2, v2, density=1.2)
axes[0, 1].set_title("Solid-Body Rotational Flow")

# Common vorticity scale
levels = np.linspace(-2.1, 2.1, 22)

# Irrotational vorticity
c1 = axes[1, 0].contourf(
    X, Y, w1_plot, levels=levels, cmap="coolwarm"
)
axes[1, 0].set_title("Irrotational Flow - Vorticity")
fig.colorbar(c1, ax=axes[1, 0], label="Vorticity (1/s)")

# Rotational vorticity
c2 = axes[1, 1].contourf(
    X, Y, w2_plot, levels=levels, cmap="coolwarm"
)
axes[1, 1].set_title("Rotational Flow - Vorticity")
fig.colorbar(c2, ax=axes[1, 1], label="Vorticity (1/s)")

for ax in axes.flat:
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.2)

plt.tight_layout()

# Save graph
Path("outputs").mkdir(exist_ok=True)
plt.savefig("outputs/flow_comparison.png", dpi=300)

print("IRROTATIONAL FLOW: Vorticity =", np.mean(w1))
print("ROTATIONAL FLOW: Vorticity =", np.mean(w2))
print("Simulation completed successfully!")

plt.show()
