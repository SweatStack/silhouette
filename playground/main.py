# /// script
# dependencies = ["numpy", "scipy", "scikit-learn", "silhouette"]
# ///
"""
Fit silhouette power-duration models to user-provided data points.

Receives JSON {durations: [...], values: [...], domain: "power"|"speed"}
via ctx.params. Returns fitted parameters and curve data via
window.onFitResults().
"""

from runtime import ctx
from pyscript import window
import json
import numpy as np

data = json.loads(str(ctx.params))
durations = np.array(data["durations"], dtype=float)
values = np.array(data["values"], dtype=float)
domain = data.get("domain", "power")
body_mass = data.get("bodyMass", 70)

X = durations.reshape(-1, 1)

# Curve evaluation range — always start at t=1, extend beyond data
t_max = min(7200, durations.max() * 2)
t_curve = np.logspace(0, np.log10(t_max), 300)

results = {}

if domain == "speed":
    from silhouette import (
        TwoParamCriticalSpeedRegressor,
        ThreeParamCriticalSpeedRegressor,
        ExpSpeedRegressor,
        OmniDomainSpeedRegressor,
        MinimalPowerSpeedRegressor,
        VDOTSpeedRegressor,
    )
    models = {
        "two_parameter": (TwoParamCriticalSpeedRegressor, 2, {}),
        "two_parameter_work": (TwoParamCriticalSpeedRegressor, 2, {"fitting": "work_duration"}),
        "three_parameter": (ThreeParamCriticalSpeedRegressor, 3, {}),
        "exponential": (ExpSpeedRegressor, 3, {}),
        "omni": (OmniDomainSpeedRegressor, 5, {}),
        "minimal_power": (MinimalPowerSpeedRegressor, 4, {}),
        "vdot": (VDOTSpeedRegressor, 2, {}),
    }
else:
    from silhouette import (
        TwoParamCriticalPowerRegressor,
        ThreeParamCriticalPowerRegressor,
        ExpPowerRegressor,
        OmniDomainPowerRegressor,
        MinimalPowerPowerRegressor,
        VDOTPowerRegressor,
    )
    models = {
        "two_parameter": (TwoParamCriticalPowerRegressor, 2, {}),
        "two_parameter_work": (TwoParamCriticalPowerRegressor, 2, {"fitting": "work_duration"}),
        "three_parameter": (ThreeParamCriticalPowerRegressor, 3, {}),
        "exponential": (ExpPowerRegressor, 3, {}),
        "omni": (OmniDomainPowerRegressor, 5, {}),
        "minimal_power": (MinimalPowerPowerRegressor, 4, {}),
        "vdot": (VDOTPowerRegressor, 2, {"body_mass": body_mass}),
    }

for name, (Model, min_points, kwargs) in models.items():
    if len(durations) < min_points:
        continue
    try:
        reg = Model(**kwargs)
        reg.fit(X, values)
        v_curve = reg.predict(t_curve.reshape(-1, 1))
        params = {}
        for p in reg._PARAM_ORDER:
            params[p] = round(float(getattr(reg, f"{p}_")), 2)
        results[name] = {
            "params": params,
            "curve": {"t": t_curve.tolist(), "v": v_curve.tolist()},
        }
    except Exception as e:
        results[name] = {"error": str(e)}

window.onFitResults(json.dumps(results))
