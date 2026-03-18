import numpy as np

from silhouette._base import BaseRegressor


def _omni_curve(t, critical, maximum, capacity, a, tcp_max):
    """Evaluate the omni-domain model.

    Parameters
    ----------
    t : array-like
        Durations in seconds.
    critical : float
        Critical power (W) or critical speed (m/s).
    maximum : float
        Peak power (W) or maximum speed (m/s).
    capacity : float
        Anaerobic work capacity (J) or distance capacity (m).
    a : float
        Fatigue factor for durations beyond tcp_max.
    tcp_max : float
        Transition duration (s) where the logarithmic fatigue term activates.

    Returns
    -------
    output : ndarray
        Predicted power (W) or speed (m/s).
    """
    t = np.asarray(t)
    base = capacity / t * (1 - np.exp(-t * (maximum - critical) / capacity)) + critical
    return np.where(t <= tcp_max, base, base - a * np.log(t / tcp_max))


class OmniDomainPowerRegressor(BaseRegressor):
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
    def curve(t, *, cp, p_max, w_prime, a, tcp_max):
        return _omni_curve(t, cp, p_max, w_prime, a, tcp_max)


class OmniDomainSpeedRegressor(BaseRegressor):
    """Omni-domain speed-duration model.

    Captures the full speed-duration relationship from short sprints to
    multi-hour efforts for running and other locomotion. This is the
    speed-domain analogue of the omni-domain power model.

    Combines exponential decay at short durations with logarithmic decay
    beyond a transition point:

        For t <= tcp_max:
            v(t) = d_prime / t * (1 - exp(-t * (s_max - cs) / d_prime)) + cs

        For t > tcp_max:
            v(t) = d_prime / t * (1 - exp(-t * (s_max - cs) / d_prime)) + cs
                   - a * ln(t / tcp_max)

    Parameters
    ----------
    bounds : dict, optional
        Parameter bounds for optimization. Keys are "cs", "s_max",
        "d_prime", "a", and "tcp_max". Values are (lower, upper) tuples.
        Missing keys fall back to defaults suitable for running speed (m/s).
    initial_params : dict, optional
        Initial parameter estimates for optimization. Same keys as bounds.
        Missing keys fall back to defaults suitable for running speed (m/s).
    method : str, default="Nelder-Mead"
        Optimization method passed to scipy.optimize.minimize.
    max_iter : int, default=10_000
        Maximum number of optimizer iterations.

    Attributes
    ----------
    cs_ : float
        Critical speed (m/s).
    s_max_ : float
        Maximum sprint speed (m/s).
    d_prime_ : float
        Distance capacity above critical speed (m).
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

    _PARAM_ORDER = ("cs", "s_max", "d_prime", "a", "tcp_max")

    _DEFAULT_BOUNDS = {
        "cs": (0.5, 10),
        "s_max": (0.5, 20),
        "d_prime": (1, 2_000),
        "a": (0, None),
        "tcp_max": (1200, 7200),
    }

    _DEFAULT_INITIAL_PARAMS = {
        "cs": 4,
        "s_max": 10,
        "d_prime": 300,
        "a": 0.5,
        "tcp_max": 1800,
    }

    _RECOMMENDED_DURATION_RANGE = None

    @staticmethod
    def curve(t, *, cs, s_max, d_prime, a, tcp_max):
        return _omni_curve(t, cs, s_max, d_prime, a, tcp_max)
