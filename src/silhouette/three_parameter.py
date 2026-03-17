import numpy as np

from silhouette._base import BaseRegressor


class ThreeParameterRegressor(BaseRegressor):
    """Three-parameter critical power model.

    Extends the two-parameter model with a maximum instantaneous power term,
    bounding predicted power at short durations:

        P(t) = (w_prime * p_max + t * cp * (p_max - cp)) / (w_prime + t * (p_max - cp))

    As t approaches 0, P(t) approaches p_max. As t approaches infinity,
    P(t) approaches cp.

    Parameters
    ----------
    bounds : dict, optional
        Parameter bounds for optimization. Keys are "cp", "w_prime",
        and "p_max".
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
    p_max_ : float
        Maximum instantaneous power (W).
    opt_result_ : scipy.optimize.OptimizeResult
        Raw optimizer result.

    References
    ----------
    Morton, R. H. (1996). A 3-parameter critical power model.
    Ergonomics, 39(4), 611-619.
    """

    _PARAM_ORDER = ("cp", "w_prime", "p_max")
    _DEFAULT_BOUNDS = {"cp": (1, 800), "w_prime": (500, 60_000), "p_max": (500, 2000)}
    _DEFAULT_INITIAL_PARAMS = {"cp": 300, "w_prime": 20_000, "p_max": 1000}

    @staticmethod
    def _model(t, cp, w_prime, p_max):
        numerator = w_prime * p_max + t * cp * (p_max - cp)
        denominator = w_prime + t * (p_max - cp)
        return numerator / denominator
