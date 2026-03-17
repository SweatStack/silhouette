import numpy as np

from silhouette._base import BaseRegressor


class OmniDurationRegressor(BaseRegressor):
    """Omni-domain power-duration model.

    Captures the full power-duration relationship from short sprints to
    multi-hour efforts. Combines exponential decay at short durations with
    logarithmic decay beyond a transition point:

        For t <= tcp_max:
            P(t) = w_prime / t * (1 - exp(-t * (p_max - cp) / w_prime)) + cp

        For t > tcp_max:
            P(t) = w_prime / t * (1 - exp(-t * (p_max - cp) / w_prime)) + cp
                   - a * ln(t / tcp_max)

    Parameters
    ----------
    bounds : dict, optional
        Parameter bounds for optimization. Keys are "cp", "p_max", "w_prime",
        "a", and "tcp_max". Values are (lower, upper) tuples. Missing keys
        fall back to defaults suitable for cycling power (watts).
    initial_params : dict, optional
        Initial parameter estimates for optimization. Same keys as bounds.
        Missing keys fall back to defaults suitable for cycling power (watts).
    method : str, default="Nelder-Mead"
        Optimization method passed to scipy.optimize.minimize.
    max_iter : int, default=10_000
        Maximum number of optimizer iterations.

    Attributes
    ----------
    cp_ : float
        Critical power (W).
    p_max_ : float
        Peak power (W).
    w_prime_ : float
        Anaerobic work capacity (J).
    a_ : float
        Fatigue factor for durations beyond tcp_max.
    tcp_max_ : float
        Transition duration (s) where the logarithmic fatigue term activates.
    opt_result_ : scipy.optimize.OptimizeResult
        Raw optimizer result.

    References
    ----------
    Puchowicz, M. J., Baker, J., & Clarke, D. C. (2020). Development and field
    validation of an omni-domain power-duration model. Journal of Sports
    Sciences, 38(7), 801-813.
    """

    _PARAM_ORDER = ("cp", "p_max", "w_prime", "a", "tcp_max")

    _DEFAULT_BOUNDS = {
        "cp": (1, 800),
        "p_max": (1, 4000),
        "w_prime": (1, 60_000),
        "a": (0, None),
        "tcp_max": (1200, 7200),
    }

    _DEFAULT_INITIAL_PARAMS = {
        "cp": 300,
        "p_max": 1000,
        "w_prime": 20_000,
        "a": 50,
        "tcp_max": 1800,
    }

    @staticmethod
    def _model(t, cp, p_max, w_prime, a, tcp_max):
        base = w_prime / t * (1 - np.exp(-t * (p_max - cp) / w_prime)) + cp
        return np.where(t <= tcp_max, base, base - a * np.log(t / tcp_max))
