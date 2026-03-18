import numpy as np

from silhouette._base import BaseRegressor


def _three_param_curve(t, critical, capacity, maximum):
    """Evaluate the three-parameter hyperbolic model.

    Parameters
    ----------
    t : array-like
        Durations in seconds.
    critical : float
        Critical power (W) or critical speed (m/s).
    capacity : float
        Anaerobic work capacity (J) or distance capacity (m).
    maximum : float
        Maximum instantaneous power (W) or speed (m/s).

    Returns
    -------
    output : ndarray
        Predicted power (W) or speed (m/s).
    """
    t = np.asarray(t)
    numerator = capacity * maximum + t * critical * (maximum - critical)
    denominator = capacity + t * (maximum - critical)
    return numerator / denominator


class ThreeParamCriticalPowerRegressor(BaseRegressor):
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
    _RECOMMENDED_DURATION_RANGE = (None, 900)

    @staticmethod
    def curve(t, *, cp, w_prime, p_max):
        return _three_param_curve(t, cp, w_prime, p_max)


class ThreeParamCriticalSpeedRegressor(BaseRegressor):
    """Three-parameter critical speed model.

    Extends the two-parameter critical speed model with a maximum
    instantaneous speed term, bounding predicted speed at short durations:

        v(t) = (d_prime * s_max + t * cs * (s_max - cs))
               / (d_prime + t * (s_max - cs))

    As t approaches 0, v(t) approaches s_max (maximum sprint speed). As t
    approaches infinity, v(t) approaches cs (critical speed).

    This is the speed-domain analogue of the three-parameter critical power
    model, suitable for running and other locomotion where speed rather than
    power is the measured output.

    Parameters
    ----------
    bounds : dict, optional
        Parameter bounds for optimization. Keys are "cs", "d_prime",
        and "s_max".
    initial_params : dict, optional
        Initial estimates. Same keys as bounds.
    method : str, default="Nelder-Mead"
        Optimization method passed to scipy.optimize.minimize.
    max_iter : int, default=10_000
        Maximum number of optimizer iterations.

    Attributes
    ----------
    cs_ : float
        Critical speed (m/s).
    d_prime_ : float
        Distance capacity above critical speed (m).
    s_max_ : float
        Maximum instantaneous speed (m/s).
    opt_result_ : scipy.optimize.OptimizeResult
        Raw optimizer result.

    References
    ----------
    Morton, R. H. (1996). A 3-parameter critical power model.
    Ergonomics, 39(4), 611-619.
    """

    _PARAM_ORDER = ("cs", "d_prime", "s_max")
    _DEFAULT_BOUNDS = {"cs": (0.5, 10), "d_prime": (20, 1_000), "s_max": (5, 20)}
    _DEFAULT_INITIAL_PARAMS = {"cs": 4, "d_prime": 300, "s_max": 10}
    _RECOMMENDED_DURATION_RANGE = (None, 900)

    @staticmethod
    def curve(t, *, cs, d_prime, s_max):
        return _three_param_curve(t, cs, d_prime, s_max)
