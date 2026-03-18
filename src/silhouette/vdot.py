import numpy as np

from silhouette._base import BaseRegressor

# Oxygen cost of cycling: 11.7 mL O2 per watt per minute (absolute).
# To get ml/kg/min: vo2 = (power * 11.7) / body_mass
_ML_O2_PER_WATT = 11.7


def _vo2_from_power(watts, body_mass):
    """Convert power (W) to VO2 (ml/kg/min).

    Parameters
    ----------
    watts : float or ndarray
        Power in watts.
    body_mass : float
        Body mass in kg.

    Returns
    -------
    vo2 : float or ndarray
        Oxygen cost in ml/kg/min.
    """
    return np.asarray(watts) * _ML_O2_PER_WATT / body_mass


def _power_from_vo2(vo2, body_mass):
    """Convert VO2 (ml/kg/min) to power (W).

    Parameters
    ----------
    vo2 : float or ndarray
        Oxygen cost in ml/kg/min.
    body_mass : float
        Body mass in kg.

    Returns
    -------
    watts : float or ndarray
        Power in watts.
    """
    return np.asarray(vo2) * body_mass / _ML_O2_PER_WATT


def _vo2_from_speed(v_ms):
    """Convert speed (m/s) to VO2 (ml/kg/min) via the Daniels-Gilbert polynomial.

    Parameters
    ----------
    v_ms : float or ndarray
        Speed in meters per second.

    Returns
    -------
    vo2 : float or ndarray
        Oxygen cost in ml/kg/min.
    """
    v_mmin = np.asarray(v_ms) * 60  # m/s -> m/min
    return -4.60 + 0.182258 * v_mmin + 0.000104 * v_mmin**2


def _speed_from_vo2(vo2):
    """Convert VO2 (ml/kg/min) to speed (m/s) via the quadratic formula.

    Solves: 0.000104*v^2 + 0.182258*v + (-4.60 - vo2) = 0 for v in m/min,
    then converts to m/s.

    Parameters
    ----------
    vo2 : float or ndarray
        Oxygen cost in ml/kg/min.

    Returns
    -------
    v_ms : float or ndarray
        Speed in meters per second.
    """
    vo2 = np.asarray(vo2)
    a = 0.000104
    b = 0.182258
    c = -4.60 - vo2
    v_mmin = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    return v_mmin / 60  # m/min -> m/s


def _fraction_vo2max(t_sec):
    """Sustainable fraction of VO2max as a function of duration.

    Parameters
    ----------
    t_sec : float or ndarray
        Duration in seconds.

    Returns
    -------
    fraction : float or ndarray
        Fraction of VO2max that can be sustained.
    """
    t = np.asarray(t_sec) / 60  # seconds -> minutes
    return (
        0.8
        + 0.1894393 * np.exp(-0.012778 * t)
        + 0.2989558 * np.exp(-0.1932605 * t)
    )


def _vdot_curve(t, vdot):
    """Evaluate the VDOT speed-duration model.

    Parameters
    ----------
    t : array-like
        Durations in seconds.
    vdot : float
        VDOT value (ml/kg/min).

    Returns
    -------
    speed : ndarray
        Predicted speed in m/s.
    """
    t = np.asarray(t)
    vo2 = vdot * _fraction_vo2max(t)
    return _speed_from_vo2(vo2)


def _vdot_power_curve(t, vdot, body_mass):
    """Evaluate the VDOT power-duration model.

    Parameters
    ----------
    t : array-like
        Durations in seconds.
    vdot : float
        VDOT value (ml/kg/min).
    body_mass : float
        Body mass in kg.

    Returns
    -------
    power : ndarray
        Predicted power in watts.
    """
    t = np.asarray(t)
    vo2 = vdot * _fraction_vo2max(t)
    return _power_from_vo2(vo2, body_mass)


class VDOTPowerRegressor(BaseRegressor):
    """VDOT (Daniels-Gilbert) power-duration model.

    An experimental adaptation of the VDOT model to cycling power. Uses the
    same sustainable-VO2-fraction curve as the running model, combined with
    a linear VO2-to-power conversion (11.7 mL O2/W).

    The f(t) duration component was originally fitted to running data. The
    sustainable fraction coefficients (0.8 asymptote, two time constants) may
    not perfectly reflect cycling physiology, where threshold fractions are
    typically higher (~85-90%). Use with caution.

        P(t) = VDOT * f(t) * body_mass / 11.7

    Parameters
    ----------
    body_mass : float
        Athlete's body mass in kg. Required for the VO2-to-power conversion.
    bounds : dict, optional
        Parameter bounds for optimization. Key is "vdot".
    initial_params : dict, optional
        Initial estimates. Same keys as bounds.
    method : str, default="Nelder-Mead"
        Optimization method passed to scipy.optimize.minimize.
    max_iter : int, default=10_000
        Maximum number of optimizer iterations.

    Attributes
    ----------
    vdot_ : float
        Fitted VDOT value (ml/kg/min).
    opt_result_ : scipy.optimize.OptimizeResult
        Raw optimizer result.

    References
    ----------
    Daniels, J., & Gilbert, J. (1979). Oxygen Power: Performance Tables for
    Distance Runners. Tempe, AZ.
    """

    _PARAM_ORDER = ("vdot",)
    _DEFAULT_BOUNDS = {"vdot": (20, 90)}
    _DEFAULT_INITIAL_PARAMS = {"vdot": 50}
    _RECOMMENDED_DURATION_RANGE = (180, 7200)

    def __init__(self, body_mass=70, duration_range=None, bounds=None,
                 initial_params=None, method="Nelder-Mead", max_iter=10_000):
        super().__init__(
            duration_range=duration_range,
            bounds=bounds,
            initial_params=initial_params,
            method=method,
            max_iter=max_iter,
        )
        self.body_mass = body_mass

    @staticmethod
    def curve(t, *, vdot, body_mass):
        return _vdot_power_curve(t, vdot, body_mass)

    def _fitted_params(self):
        params = super()._fitted_params()
        params["body_mass"] = self.body_mass
        return params

    def fit(self, X, y):
        """Fit the model to duration-power data.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Durations in seconds.
        y : array-like of shape (n_samples,)
            Power in watts.

        Returns
        -------
        self
        """
        X, y = self._preprocess_data(X, y)
        t = X[:, 0]

        def objective(values):
            params = dict(zip(self._PARAM_ORDER, values))
            return self._cost_function(
                self.curve(t, body_mass=self.body_mass, **params), y,
            )

        from scipy.optimize import minimize

        result = minimize(
            objective,
            self._resolve_initial_params(),
            method=self.method,
            bounds=self._resolve_bounds(),
            options={"maxiter": self.max_iter},
        )

        for name, value in zip(self._PARAM_ORDER, result.x):
            setattr(self, f"{name}_", value)

        self.opt_result_ = result
        self.is_fitted_ = True

        return self


class VDOTSpeedRegressor(BaseRegressor):
    """VDOT (Daniels-Gilbert) speed-duration model.

    A single-parameter model that predicts running speed from duration using
    the VDOT fitness metric. The model combines an empirical oxygen-cost-vs-
    velocity polynomial with a sustainable-VO2-fraction-vs-duration double
    exponential:

        VDOT = VO2(v) / f(t)

    where:
        VO2(v) = -4.60 + 0.182258*v + 0.000104*v^2  (v in m/min)
        f(t) = 0.8 + 0.1894393*e^(-0.012778*t) + 0.2989558*e^(-0.1932605*t)
               (t in minutes)

    Given a VDOT value, the model solves for speed at each duration.

    Parameters
    ----------
    bounds : dict, optional
        Parameter bounds for optimization. Key is "vdot".
    initial_params : dict, optional
        Initial estimates. Same keys as bounds.
    method : str, default="Nelder-Mead"
        Optimization method passed to scipy.optimize.minimize.
    max_iter : int, default=10_000
        Maximum number of optimizer iterations.

    Attributes
    ----------
    vdot_ : float
        Fitted VDOT value (ml/kg/min).
    opt_result_ : scipy.optimize.OptimizeResult
        Raw optimizer result.

    References
    ----------
    Daniels, J., & Gilbert, J. (1979). Oxygen Power: Performance Tables for
    Distance Runners. Tempe, AZ.
    """

    _PARAM_ORDER = ("vdot",)
    _DEFAULT_BOUNDS = {"vdot": (20, 90)}
    _DEFAULT_INITIAL_PARAMS = {"vdot": 50}
    _RECOMMENDED_DURATION_RANGE = (180, 7200)

    @staticmethod
    def curve(t, *, vdot):
        return _vdot_curve(t, vdot)
