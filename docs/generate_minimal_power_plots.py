# /// script
# dependencies = ["silhouette", "matplotlib", "numpy"]
# ///
"""Generate example MinimalPowerDisplay plots for power and speed."""

import matplotlib.pyplot as plt
import numpy as np

from silhouette import MinimalPowerPowerRegressor, MinimalPowerSpeedRegressor
from silhouette.plotting import MinimalPowerDisplay

# --- Power (cycling) ---

durations = np.array([60, 120, 300, 600, 1200, 2700])
power = np.array([480, 400, 340, 310, 285, 260])
X = durations.reshape(-1, 1)

reg_power = MinimalPowerPowerRegressor()
reg_power.fit(X, power)

fig, ax = plt.subplots(figsize=(8, 5))
MinimalPowerDisplay.from_estimator(reg_power, X, power, ax=ax)
ax.set_title(
    f"MAP={reg_power.map_:.0f}W, "
    f"MAP duration={reg_power.map_duration_:.0f}s, "
    f"$\\gamma_l$={reg_power.gamma_l_:.3f}, $\\gamma_s$={reg_power.gamma_s_:.3f}"
)
plt.tight_layout()
fig.savefig("docs/minimal_power_power.png", dpi=150)
print("Saved docs/minimal_power_power.png")

# --- Speed (running) ---

durations = np.array([60, 120, 300, 600, 1200, 2700])
speed = np.array([6.2, 5.6, 5.0, 4.6, 4.2, 3.9])
X = durations.reshape(-1, 1)

reg_speed = MinimalPowerSpeedRegressor()
reg_speed.fit(X, speed)

fig, ax = plt.subplots(figsize=(8, 5))
MinimalPowerDisplay.from_estimator(reg_speed, X, speed, ax=ax)
ax.set_title(
    f"MAS={reg_speed.map_:.2f}m/s, "
    f"MAS duration={reg_speed.map_duration_:.0f}s, "
    f"$\\gamma_l$={reg_speed.gamma_l_:.3f}, $\\gamma_s$={reg_speed.gamma_s_:.3f}"
)
plt.tight_layout()
fig.savefig("docs/minimal_power_speed.png", dpi=150)
print("Saved docs/minimal_power_speed.png")
