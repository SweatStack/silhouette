# Silhouette

Silhouette is a Python library for intensity-duration modelling in cycling. It provides scikit-learn compatible regressors for fitting power-duration curves to cycling data.

The first model available is the omni-domain power-duration model from Puchowicz et al. (2020), which captures the full power-duration relationship from short sprints to long endurance efforts using five physiologically meaningful parameters: critical power (CP), peak power (P_max), anaerobic work capacity (W'), a fatigue factor, and a transition duration.

## Installation

```bash
uv add silhouette
```

Or with pip:

```bash
pip install silhouette
```

## Usage

```python
import numpy as np
from silhouette import OmniDurationRegressor

# Durations in seconds and corresponding power in watts
durations = np.array([5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600])
power = np.array([1050, 850, 600, 480, 400, 340, 310, 290, 275, 255])

reg = OmniDurationRegressor()
reg.fit(durations.reshape(-1, 1), power)

# Fitted parameters
print(f"CP: {reg.cp_:.0f} W")
print(f"P_max: {reg.p_max_:.0f} W")
print(f"W': {reg.w_prime_:.0f} J")

# Predict power at any duration
predicted = reg.predict(np.array([[60], [300], [3600]]))
print(f"1 min: {predicted[0]:.0f} W, 5 min: {predicted[1]:.0f} W, 60 min: {predicted[2]:.0f} W")
```

## References

Puchowicz, M. J., Baker, J., & Clarke, D. C. (2020). Development and field validation of an omni-domain power-duration model. Journal of Sports Sciences, 38(7), 801-813.
