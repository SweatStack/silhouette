# Silhouette

A Python library for fitting power-duration curves to cycling data. Scikit-learn compatible.

## Models

| Model | Parameters |
|---|---|
| `TwoParameterRegressor` | CP, W' |
| `ThreeParameterRegressor` | CP, W', P_max |
| `OmniDurationRegressor` | CP, W', P_max, a, tcp_max |

## Installation

```bash
uv add silhouette
```

Or with pip:

```bash
pip install silhouette
```

## Quick start

```python
import numpy as np
from silhouette import OmniDurationRegressor

durations = np.array([5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600])
power = np.array([1050, 850, 600, 480, 400, 340, 310, 290, 275, 255])

reg = OmniDurationRegressor()
reg.fit(durations.reshape(-1, 1), power)

reg.cp_       # critical power (W)
reg.p_max_    # peak power (W)
reg.w_prime_  # anaerobic work capacity (J)

reg.predict(np.array([[300]]))  # predicted power at 5 minutes
```

All three models share the same interface. Swap `OmniDurationRegressor` for `TwoParameterRegressor` or `ThreeParameterRegressor` and the code works the same way.

## Custom bounds

```python
reg = OmniDurationRegressor(
    bounds={"cp": (200, 400), "p_max": (800, 1500)},
    initial_params={"cp": 280},
)
```

## Time to exhaustion

```python
power, tte = reg.predict_inverse()
# power: array of watt values
# tte: corresponding time to exhaustion in seconds
```

## References

- Monod, H., & Scherrer, J. (1965). The work capacity of a synergic muscular group. *Ergonomics, 8*(3), 329-338.
- Morton, R. H. (1996). A 3-parameter critical power model. *Ergonomics, 39*(4), 611-619.
- Puchowicz, M. J., Baker, J., & Clarke, D. C. (2020). Development and field validation of an omni-domain power-duration model. *Journal of Sports Sciences, 38*(7), 801-813.
