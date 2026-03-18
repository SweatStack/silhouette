# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "silhouette[plotting]",
#     "numpy",
# ]
# ///
"""Generate plots for the README.

Run from the project root:
    uv run docs/generate_plots.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from silhouette import TwoParameterRegressor, ThreeParameterRegressor, OmniDurationRegressor, FPCARegressor
from silhouette.plotting import PowerDurationDisplay, ModeOfVarianceDisplay

DOCS_DIR = Path(__file__).parent

durations = np.array([5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600])
power = np.array([1050, 850, 600, 480, 400, 340, 310, 290, 275, 255])

X = durations.reshape(-1, 1)

reg_2p = TwoParameterRegressor().fit(X, power)
reg_3p = ThreeParameterRegressor().fit(X, power)
reg_omni = OmniDurationRegressor().fit(X, power)

fig, ax = plt.subplots(figsize=(10, 5))
display = PowerDurationDisplay.from_estimators(
    [reg_2p, reg_3p, reg_omni],
    X, power,
    names=["2-parameter", "3-parameter", "Omni"],
    ax=ax,
)
display.figure_.savefig(DOCS_DIR / "power_duration.png", dpi=150, bbox_inches="tight")
print(f"Saved {DOCS_DIR / 'power_duration.png'}")

# FPCA mode of variance
fpca_reg = FPCARegressor.from_model().fit(X, power)
display = ModeOfVarianceDisplay.from_estimator(fpca_reg)
display.figure_.savefig(DOCS_DIR / "mode_of_variance.png", dpi=150, bbox_inches="tight")
print(f"Saved {DOCS_DIR / 'mode_of_variance.png'}")
