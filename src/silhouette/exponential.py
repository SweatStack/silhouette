import numpy as np

from silhouette._base import BaseRegressor


def _exp_curve(t, critical, maximum, tau):
    """Evaluate the exponential power-duration model.

    Parameters
    ----------
    t : array-like
        Durations in seconds.
    critical : float
        Critical power (W) or critical speed (m/s).
    maximum : float
        Maximum instantaneous power (W) or speed (m/s).
    tau : float
        Time constant (s) governing the decay rate.

    Returns
    -------
    output : ndarray
        Predicted power (W) or speed (m/s).
    """
    t = np.asarray(t)
    return (maximum - critical) * np.exp(-t / tau) + critical


class ExpPowerRegressor(BaseRegressor):
    """Exponential critical power model.

    A three-parameter model that bounds power at short durations using an
    exponential decay from maximum power toward critical power:

        P(t) = (p_max - cp) * exp(-t / tau) + cp

    As t approaches 0, P(t) approaches p_max. As t approaches infinity,
    P(t) approaches cp.

    Parameters
    ----------
    bounds : dict, optional
        Parameter bounds for optimization. Keys are "cp", "p_max", and "tau".
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
    p_max_ : float
        Maximum instantaneous power (W).
    tau_ : float
        Time constant (s).
    opt_result_ : scipy.optimize.OptimizeResult
        Raw optimizer result.

    References
    ----------
    Hopkins, W. G., Edmond, I. M., Hamilton, B. H., Macfarlane, D. J., &
    Ross, B. H. (1989). Relation between power and endurance for treadmill
    running and cycle ergometry. Canadian Journal of Sport Sciences.
    """

    _PARAM_ORDER = ("cp", "p_max", "tau")
    _DEFAULT_BOUNDS = {"cp": (1, 800), "p_max": (500, 2000), "tau": (1, 600)}
    _DEFAULT_INITIAL_PARAMS = {"cp": 300, "p_max": 1000, "tau": 60}
    _RECOMMENDED_DURATION_RANGE = (None, 900)

    @staticmethod
    def curve(t, *, cp, p_max, tau):
        return _exp_curve(t, cp, p_max, tau)


class ExpSpeedRegressor(BaseRegressor):
    """Exponential critical speed model.

    A three-parameter model that bounds speed at short durations using an
    exponential decay from maximum speed toward critical speed:

        v(t) = (s_max - cs) * exp(-t / tau) + cs

    As t approaches 0, v(t) approaches s_max. As t approaches infinity,
    v(t) approaches cs.

    Parameters
    ----------
    bounds : dict, optional
        Parameter bounds for optimization. Keys are "cs", "s_max", and "tau".
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
    s_max_ : float
        Maximum instantaneous speed (m/s).
    tau_ : float
        Time constant (s).
    opt_result_ : scipy.optimize.OptimizeResult
        Raw optimizer result.

    References
    ----------
    Hopkins, W. G., Edmond, I. M., Hamilton, B. H., Macfarlane, D. J., &
    Ross, B. H. (1989). Relation between power and endurance for treadmill
    running and cycle ergometry. Canadian Journal of Sport Sciences.
    """

    _PARAM_ORDER = ("cs", "s_max", "tau")
    _DEFAULT_BOUNDS = {"cs": (0.5, 10), "s_max": (5, 20), "tau": (1, 600)}
    _DEFAULT_INITIAL_PARAMS = {"cs": 4, "s_max": 10, "tau": 60}
    _RECOMMENDED_DURATION_RANGE = (None, 900)

    @staticmethod
    def curve(t, *, cs, s_max, tau):
        return _exp_curve(t, cs, s_max, tau)
