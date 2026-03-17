# Silhouette

A Python library for fitting power-duration curves to cycling data. Scikit-learn compatible.

## Models

### Parametric models

| Model | Parameters |
|---|---|
| `TwoParameterRegressor` | CP, W' |
| `ThreeParameterRegressor` | CP, W', P_max |
| `OmniDurationRegressor` | CP, W', P_max, a, tcp_max |

### Data-driven models

| Model | Parameters |
|---|---|
| `FPCARegressor` | FPC1 (gain), FPC2 (sprint/endurance bias), FPC3 (mid-duration) |

## Installation

```bash
uv add silhouette
```

Or with pip:

```bash
pip install silhouette
```

## Quick start

### Parametric models

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

All parametric models share the same interface. Swap `OmniDurationRegressor` for `TwoParameterRegressor` or `ThreeParameterRegressor` and the code works the same way.

### FPCA model

```python
from silhouette import FPCARegressor

reg = FPCARegressor.from_model()
reg.fit(durations.reshape(-1, 1), power)

reg.fpc1_     # overall power level
reg.fpc2_     # sprint vs endurance bias
reg.fpc3_     # mid-duration specialization

reg.predict(np.array([[300]]))
reg.percentiles()  # {"fpc1": 72.3, "fpc2": 34.1, "fpc3": 55.8}
reg.z_scores()     # {"fpc1": 0.87, "fpc2": -0.41, "fpc3": 0.14}
```

## Known parameters

When parameters are already known, use `curve` directly without fitting:

```python
from silhouette import TwoParameterRegressor, FPCARegressor

t = np.arange(1, 3601)
power = TwoParameterRegressor.curve(t, cp=250, w_prime=20_000)
power = FPCARegressor.curve(t, fpc1=0.5, fpc2=-0.1, fpc3=0.0)
```

## Custom bounds

```python
reg = OmniDurationRegressor(
    bounds={"cp": (200, 400), "p_max": (800, 1500)},
    initial_params={"cp": 280},
)
```

## Time to exhaustion

The inverse of the power-duration curve: given a power, how long can it be sustained?

```python
# On a fitted model
tte = reg.predict_inverse(np.array([250, 300, 350]))

# With known parameters
tte = TwoParameterRegressor.curve_inverse(350, cp=250, w_prime=20_000)
```

## References

- Monod, H., & Scherrer, J. (1965). The work capacity of a synergic muscular group. *Ergonomics, 8*(3), 329-338.
- Morton, R. H. (1996). A 3-parameter critical power model. *Ergonomics, 39*(4), 611-619.
- Puchowicz, M. J., Baker, J., & Clarke, D. C. (2020). Development and field validation of an omni-domain power-duration model. *Journal of Sports Sciences, 38*(7), 801-813.
- Puchowicz, M. J., & Skiba, P. F. (2025). Functional Data Analysis of the Power-Duration Relationship in Cyclists. *International Journal of Sports Physiology and Performance, 1*(aop), 1-10.
