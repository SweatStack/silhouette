import numpy as np

from silhouette._base import BaseRegressor


class TwoParameterRegressor(BaseRegressor):
    """Two-parameter critical power model.

    The classic hyperbolic relationship between power and duration:

        P(t) = cp + w_prime / t

    Best suited for durations between 2 and 20 minutes.

    Parameters
    ----------
    bounds : dict, optional
        Parameter bounds for optimization. Keys are "cp" and "w_prime".
    initial_params : dict, optional
        Initial estimates. Same keys as bounds.
    method : str, default="Nelder-Mead"
        Optimization method passed to scipy.optimize.minimize.
    max_iter : int, default=10_000
        Maximum number of optimizer iterations.

    Attributes
    ----------
    cp_ : float
        Critical power (W).
    w_prime_ : float
        Anaerobic work capacity (J).
    opt_result_ : scipy.optimize.OptimizeResult
        Raw optimizer result.

    References
    ----------
    Monod, H., & Scherrer, J. (1965). The work capacity of a synergic
    muscular group. Ergonomics, 8(3), 329-338.
    """

    _PARAM_ORDER = ("cp", "w_prime")
    _DEFAULT_BOUNDS = {"cp": (1, 800), "w_prime": (500, 40_000)}
    _DEFAULT_INITIAL_PARAMS = {"cp": 300, "w_prime": 20_000}

    @staticmethod
    def _model(t, cp, w_prime):
        return cp + w_prime / t
