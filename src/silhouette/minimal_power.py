import numpy as np
from scipy.special import lambertw

from silhouette._base import BaseRegressor


def _minimal_power_tte(work, map_val, map_duration, gamma_l, gamma_s):
    """Time to exhaustion given total work.

    Uses the W_{-1} branch of the Lambert W function to compute how long
    a given amount of work can be sustained.

    Parameters
    ----------
    work : array-like
        Total work in joules (power) or metres (speed).
    map_val : float
        Maximal aerobic power (W) or speed (m/s).
    map_duration : float
        Duration at which MAP/MAS is achieved (s).
    gamma_l : float
        Endurance parameter for long durations.
    gamma_s : float
        Endurance parameter for short durations.

    Returns
    -------
    t : ndarray
        Predicted time to exhaustion (s).
    """
    work = np.asarray(work, dtype=float)
    map_work = map_val * map_duration
    gamma = np.where(work < map_work, gamma_s, gamma_l)
    W_val = lambertw((-work / (map_work * gamma)) * np.exp(-1 / gamma), k=-1).real
    return -(work / (gamma * map_val)) / W_val


def _minimal_power_curve(t, map_val, map_duration, gamma_l, gamma_s):
    """Predict power/speed at given durations via interpolation.

    The model naturally predicts duration given work. To get
    power given duration, we generate a dense work-to-duration
    mapping and interpolate.

    Parameters
    ----------
    t : array-like
        Durations in seconds.
    map_val : float
        Maximal aerobic power (W) or speed (m/s).
    map_duration : float
        Duration at which MAP/MAS is achieved (s).
    gamma_l : float
        Endurance parameter for long durations.
    gamma_s : float
        Endurance parameter for short durations.

    Returns
    -------
    intensity : ndarray
        Predicted power (W) or speed (m/s).
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)

    # Generate dense work range for interpolation
    max_work = np.max(t) * map_val * 2
    work_range = np.linspace(1, max_work, 5000)

    t_pred = _minimal_power_tte(work_range, map_val, map_duration, gamma_l, gamma_s)
    p_pred = work_range / t_pred

    result = np.interp(t, t_pred, p_pred)
    return float(result[0]) if scalar else result


class MinimalPowerPowerRegressor(BaseRegressor):
    """Minimal power model for cycling.

    Uses the Lambert W function to model the power-duration relationship
    across aerobic and anaerobic regimes. The model splits behaviour at a
    transition point defined by MAP (maximal aerobic power) and its
    associated duration:

    - Below the transition: short-duration endurance parameter (gamma_s)
    - Above the transition: long-duration endurance parameter (gamma_l)

    This model is particularly suited for capturing the full power-duration
    curve from short efforts (~15 s) to prolonged endurance (~45 min).

    Parameters
    ----------
    duration_range : tuple of (float, float), optional
        Duration range ``(min_seconds, max_seconds)`` to use for fitting.
        If not set, all data is used but a warning is issued when data
        falls outside the recommended range of 1 min and above.
    bounds : dict, optional
        Parameter bounds for optimization. Keys are "map", "map_duration",
        "gamma_l", and "gamma_s".
    initial_params : dict, optional
        Initial estimates. Same keys as bounds.
    method : str, default="Nelder-Mead"
        Optimization method passed to scipy.optimize.minimize.
    max_iter : int, default=10_000
        Maximum number of optimizer iterations.

    Attributes
    ----------
    map_ : float
        Maximal aerobic power (W).
    map_duration_ : float
        Duration at which MAP is achieved (s).
    gamma_l_ : float
        Endurance parameter for long durations (dimensionless).
    gamma_s_ : float
        Endurance parameter for short durations (dimensionless).
    duration_mask_ : ndarray of bool
        Boolean mask indicating which training samples were used.
    opt_result_ : scipy.optimize.OptimizeResult
        Raw optimizer result.

    References
    ----------
    Mulligan, M., Adam, G., & Emig, T. (2018). A minimal power model for
    human running performance. PloS one, 13(11), e0206645.
    """

    _PARAM_ORDER = ("map", "map_duration", "gamma_l", "gamma_s")
    _DEFAULT_BOUNDS = {
        "map": (100, 800),
        "map_duration": (181, 720),
        "gamma_l": (0.01, 1.0),
        "gamma_s": (0.01, 1.0),
    }
    _DEFAULT_INITIAL_PARAMS = {
        "map": 400,
        "map_duration": 300,
        "gamma_l": 0.06,
        "gamma_s": 0.1,
    }
    _RECOMMENDED_DURATION_RANGE = (60, None)

    @staticmethod
    def curve(t, *, map, map_duration, gamma_l, gamma_s):
        return _minimal_power_curve(t, map, map_duration, gamma_l, gamma_s)

    @classmethod
    def curve_inverse(cls, y, **params):
        """Time to exhaustion for given power outputs.

        Parameters
        ----------
        y : array-like
            Power values in watts.
        **params
            Model parameters: map, map_duration, gamma_l, gamma_s.

        Returns
        -------
        tte : ndarray or float
            Time to exhaustion in seconds.
        """
        y = np.asarray(y, dtype=float)
        scalar = y.ndim == 0
        y = np.atleast_1d(y)

        t_range = np.logspace(0, np.log10(2700 * 2), 5000)
        p_range = cls.curve(t_range, **params)

        # p_range is decreasing; np.interp needs ascending xp
        # Flip both so power is ascending, durations descending
        tte = np.interp(y, p_range[::-1], t_range[::-1])
        return float(tte[0]) if scalar else tte


class MinimalPowerSpeedRegressor(BaseRegressor):
    """Minimal power model for running (speed-based).

    Uses the Lambert W function to model the speed-duration relationship
    across aerobic and anaerobic regimes. This is the speed-domain analogue
    of MinimalPowerPowerRegressor.

    The model splits behaviour at a transition point defined by MAS
    (maximal aerobic speed) and its associated duration:

    - Below the transition: short-duration endurance parameter (gamma_s)
    - Above the transition: long-duration endurance parameter (gamma_l)

    Parameters
    ----------
    duration_range : tuple of (float, float), optional
        Duration range ``(min_seconds, max_seconds)`` to use for fitting.
        If not set, all data is used but a warning is issued when data
        falls outside the recommended range of 1 min and above.
    bounds : dict, optional
        Parameter bounds for optimization. Keys are "map", "map_duration",
        "gamma_l", and "gamma_s".
    initial_params : dict, optional
        Initial estimates. Same keys as bounds.
    method : str, default="Nelder-Mead"
        Optimization method passed to scipy.optimize.minimize.
    max_iter : int, default=10_000
        Maximum number of optimizer iterations.

    Attributes
    ----------
    map_ : float
        Maximal aerobic speed (m/s).
    map_duration_ : float
        Duration at which MAS is achieved (s).
    gamma_l_ : float
        Endurance parameter for long durations (dimensionless).
    gamma_s_ : float
        Endurance parameter for short durations (dimensionless).
    duration_mask_ : ndarray of bool
        Boolean mask indicating which training samples were used.
    opt_result_ : scipy.optimize.OptimizeResult
        Raw optimizer result.

    References
    ----------
    Mulligan, M., Adam, G., & Emig, T. (2018). A minimal power model for
    human running performance. PloS one, 13(11), e0206645.
    """

    _PARAM_ORDER = ("map", "map_duration", "gamma_l", "gamma_s")
    _DEFAULT_BOUNDS = {
        "map": (0.5, 10),
        "map_duration": (181, 720),
        "gamma_l": (0.01, 1.0),
        "gamma_s": (0.01, 1.0),
    }
    _DEFAULT_INITIAL_PARAMS = {
        "map": 5,
        "map_duration": 300,
        "gamma_l": 0.06,
        "gamma_s": 0.1,
    }
    _RECOMMENDED_DURATION_RANGE = (60, None)

    @staticmethod
    def curve(t, *, map, map_duration, gamma_l, gamma_s):
        return _minimal_power_curve(t, map, map_duration, gamma_l, gamma_s)

    @classmethod
    def curve_inverse(cls, y, **params):
        """Time to exhaustion for given speed outputs.

        Parameters
        ----------
        y : array-like
            Speed values in m/s.
        **params
            Model parameters: map, map_duration, gamma_l, gamma_s.

        Returns
        -------
        tte : ndarray or float
            Time to exhaustion in seconds.
        """
        y = np.asarray(y, dtype=float)
        scalar = y.ndim == 0
        y = np.atleast_1d(y)

        t_range = np.logspace(0, np.log10(2700 * 2), 5000)
        p_range = cls.curve(t_range, **params)

        tte = np.interp(y, p_range[::-1], t_range[::-1])
        return float(tte[0]) if scalar else tte
