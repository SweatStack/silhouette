# /// script
# dependencies = ["numpy", "scipy", "scikit-learn", "silhouette"]
# ///
"""
Fit silhouette power-duration models to user-provided data points.

Receives JSON {durations: [...], powers: [...]} via ctx.params.
Returns fitted parameters and curve data via window.onFitResults().
"""

from runtime import ctx
from pyscript import window
import json
import numpy as np

data = json.loads(str(ctx.params))
durations = np.array(data["durations"], dtype=float)
powers = np.array(data["powers"], dtype=float)

X = durations.reshape(-1, 1)

# Curve evaluation range — always start at t=1, extend beyond data
t_max = min(7200, durations.max() * 2)
t_curve = np.logspace(0, np.log10(t_max), 300)

results = {}

from silhouette import (
    TwoParameterRegressor,
    ThreeParameterRegressor,
    OmniDurationRegressor,
)

models = {
    "two_parameter": (TwoParameterRegressor, 2),
    "three_parameter": (ThreeParameterRegressor, 3),
    "omni": (OmniDurationRegressor, 5),
}

for name, (Model, min_points) in models.items():
    if len(durations) < min_points:
        continue
    try:
        reg = Model()
        reg.fit(X, powers)
        p_curve = reg.predict(t_curve.reshape(-1, 1))
        params = {}
        for p in reg._PARAM_ORDER:
            params[p] = round(float(getattr(reg, f"{p}_")), 2)
        results[name] = {
            "params": params,
            "curve": {"t": t_curve.tolist(), "p": p_curve.tolist()},
        }
    except Exception as e:
        results[name] = {"error": str(e)}

window.onFitResults(json.dumps(results))
