import warnings

import numpy as np

from silhouette._base import BaseRegressor


def _two_param_curve(t, critical, capacity):
    """Evaluate the two-parameter hyperbolic model.

    Parameters
    ----------
    t : array-like
        Durations in seconds.
    critical : float
        Critical power (W) or critical speed (m/s).
    capacity : float
        Anaerobic work capacity (J) or distance capacity (m).

    Returns
    -------
    output : ndarray
        Predicted power (W) or speed (m/s).
    """
    t = np.asarray(t)
    return critical + capacity / t


class TwoParamCriticalPowerRegressor(BaseRegressor):
    """Two-parameter critical power model.

    The classic hyperbolic relationship between power and duration:

        P(t) = cp + w_prime / t

    Best suited for durations between 2 and 20 minutes.

    Two fitting approaches are available, selected via the ``fitting``
    parameter:

    - ``"nonlinear"`` (default): minimizes the sum of squared errors in
      power space using ``scipy.optimize.minimize``. This is the standard
      nonlinear least-squares approach.

    - ``"work_duration"``: linearizes the model by transforming to
      work-duration space. The identity P = CP + W'/t is rearranged to
      W = W' + CP * t, where W = P * t is the total work done. A simple
      linear regression (``np.polyfit``) is then used to estimate CP (slope)
      and W' (intercept). This minimizes the sum of squared errors in work
      space (sum of (W_pred - W_obs)^2), which gives more weight to longer
      durations since W = P * t grows with time.

    The two approaches will generally produce different parameter estimates
    for the same data.

    Parameters
    ----------
    fitting : str, default="nonlinear"
        Fitting method. One of ``"nonlinear"`` or ``"work_duration"``.
    duration_range : tuple of (float, float), optional
        Duration range ``(min_seconds, max_seconds)`` to use for fitting.
        Data points outside this range are excluded from the fit but
        ``predict()`` still works at any duration. Either bound can be
        ``None`` for one-sided filtering. If not set, all data is used
        but a warning is issued when data falls outside the recommended
        range of 2 to 15 minutes.
    bounds : dict, optional
        Parameter bounds for optimization. Keys are "cp" and "w_prime".
        Ignored when ``fitting="work_duration"``.
    initial_params : dict, optional
        Initial estimates. Same keys as bounds.
        Ignored when ``fitting="work_duration"``.
    method : str, default="Nelder-Mead"
        Optimization method passed to scipy.optimize.minimize.
        Ignored when ``fitting="work_duration"``.
    max_iter : int, default=10_000
        Maximum number of optimizer iterations.
        Ignored when ``fitting="work_duration"``.

    Attributes
    ----------
    cp_ : float
        Critical power (W).
    w_prime_ : float
        Anaerobic work capacity (J).
    duration_mask_ : ndarray of bool
        Boolean mask indicating which training samples were used after
        applying ``duration_range``.
    opt_result_ : scipy.optimize.OptimizeResult or None
        Raw optimizer result. ``None`` when ``fitting="work_duration"``.

    References
    ----------
    Monod, H., & Scherrer, J. (1965). The work capacity of a synergic
    muscular group. Ergonomics, 8(3), 329-338.
    """

    _PARAM_ORDER = ("cp", "w_prime")
    _DEFAULT_BOUNDS = {"cp": (1, 800), "w_prime": (500, 40_000)}
    _DEFAULT_INITIAL_PARAMS = {"cp": 300, "w_prime": 20_000}
    _RECOMMENDED_DURATION_RANGE = (120, 900)

    def __init__(self, fitting="nonlinear", duration_range=None, bounds=None,
                 initial_params=None, method="Nelder-Mead", max_iter=10_000):
        super().__init__(duration_range=duration_range, bounds=bounds,
                         initial_params=initial_params, method=method,
                         max_iter=max_iter)
        self.fitting = fitting

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
        valid_fitting = ("nonlinear", "work_duration")
        if self.fitting not in valid_fitting:
            raise ValueError(
                f"Invalid fitting={self.fitting!r}. "
                f"Expected one of {valid_fitting}."
            )

        if self.fitting == "nonlinear":
            return super().fit(X, y)

        # work_duration fitting
        if self.bounds is not None:
            warnings.warn(
                "bounds is ignored when fitting='work_duration'.",
                UserWarning,
                stacklevel=2,
            )
        if self.initial_params is not None:
            warnings.warn(
                "initial_params is ignored when fitting='work_duration'.",
                UserWarning,
                stacklevel=2,
            )

        X, y = self._preprocess_data(X, y)
        t = X[:, 0]
        W = y * t  # total work = power * time

        # W = CP * t + W'  =>  polyfit(t, W, 1) gives [CP, W']
        coeffs = np.polyfit(t, W, 1)
        self.cp_ = coeffs[0]
        self.w_prime_ = coeffs[1]
        self.opt_result_ = None
        self.is_fitted_ = True

        return self

    @staticmethod
    def curve(t, *, cp, w_prime):
        return _two_param_curve(t, cp, w_prime)


class TwoParamCriticalSpeedRegressor(BaseRegressor):
    """Two-parameter critical speed model.

    The hyperbolic relationship between speed and duration for running:

        v(t) = cs + d_prime / t

    This is the speed-domain analogue of the two-parameter critical power
    model. Critical speed (cs) represents the highest speed sustainable
    without progressive fatigue, and d_prime represents a finite distance
    capacity available above critical speed.

    Best suited for durations between 2 and 15 minutes (e.g., 800 m to
    5000 m race efforts).

    Two fitting approaches are available, selected via the ``fitting``
    parameter:

    - ``"nonlinear"`` (default): minimizes the sum of squared errors in
      speed space using ``scipy.optimize.minimize``.

    - ``"work_duration"``: linearizes the model by transforming to
      distance-duration space. The identity v = cs + d'/t is rearranged
      to d = d' + cs * t, where d = v * t is the total distance covered.
      A simple linear regression (``np.polyfit``) is then used to estimate
      cs (slope) and d' (intercept). This minimizes the sum of squared
      errors in distance space, giving more weight to longer durations.

    The two approaches will generally produce different parameter estimates
    for the same data.

    Parameters
    ----------
    fitting : str, default="nonlinear"
        Fitting method. One of ``"nonlinear"`` or ``"work_duration"``.
    duration_range : tuple of (float, float), optional
        Duration range ``(min_seconds, max_seconds)`` to use for fitting.
        Data points outside this range are excluded from the fit but
        ``predict()`` still works at any duration. Either bound can be
        ``None`` for one-sided filtering. If not set, all data is used
        but a warning is issued when data falls outside the recommended
        range of 2 to 15 minutes.
    bounds : dict, optional
        Parameter bounds for optimization. Keys are "cs" and "d_prime".
        Ignored when ``fitting="work_duration"``.
    initial_params : dict, optional
        Initial estimates. Same keys as bounds.
        Ignored when ``fitting="work_duration"``.
    method : str, default="Nelder-Mead"
        Optimization method passed to scipy.optimize.minimize.
        Ignored when ``fitting="work_duration"``.
    max_iter : int, default=10_000
        Maximum number of optimizer iterations.
        Ignored when ``fitting="work_duration"``.

    Attributes
    ----------
    cs_ : float
        Critical speed (m/s).
    d_prime_ : float
        Distance capacity above critical speed (m).
    duration_mask_ : ndarray of bool
        Boolean mask indicating which training samples were used after
        applying ``duration_range``.
    opt_result_ : scipy.optimize.OptimizeResult or None
        Raw optimizer result. ``None`` when ``fitting="work_duration"``.

    References
    ----------
    Monod, H., & Scherrer, J. (1965). The work capacity of a synergic
    muscular group. Ergonomics, 8(3), 329-338.
    """

    _PARAM_ORDER = ("cs", "d_prime")
    _DEFAULT_BOUNDS = {"cs": (0.5, 8), "d_prime": (20, 1_000)}
    _DEFAULT_INITIAL_PARAMS = {"cs": 4, "d_prime": 200}
    _RECOMMENDED_DURATION_RANGE = (120, 900)

    def __init__(self, fitting="nonlinear", duration_range=None, bounds=None,
                 initial_params=None, method="Nelder-Mead", max_iter=10_000):
        super().__init__(duration_range=duration_range, bounds=bounds,
                         initial_params=initial_params, method=method,
                         max_iter=max_iter)
        self.fitting = fitting

    def fit(self, X, y):
        """Fit the model to duration-speed data.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Durations in seconds.
        y : array-like of shape (n_samples,)
            Speed in metres per second.

        Returns
        -------
        self
        """
        valid_fitting = ("nonlinear", "work_duration")
        if self.fitting not in valid_fitting:
            raise ValueError(
                f"Invalid fitting={self.fitting!r}. "
                f"Expected one of {valid_fitting}."
            )

        if self.fitting == "nonlinear":
            return super().fit(X, y)

        # work_duration (distance-duration) fitting
        if self.bounds is not None:
            warnings.warn(
                "bounds is ignored when fitting='work_duration'.",
                UserWarning,
                stacklevel=2,
            )
        if self.initial_params is not None:
            warnings.warn(
                "initial_params is ignored when fitting='work_duration'.",
                UserWarning,
                stacklevel=2,
            )

        X, y = self._preprocess_data(X, y)
        t = X[:, 0]
        d = y * t  # total distance = speed * time

        # d = cs * t + d'  =>  polyfit(t, d, 1) gives [cs, d']
        coeffs = np.polyfit(t, d, 1)
        self.cs_ = coeffs[0]
        self.d_prime_ = coeffs[1]
        self.opt_result_ = None
        self.is_fitted_ = True

        return self

    @staticmethod
    def curve(t, *, cs, d_prime):
        return _two_param_curve(t, cs, d_prime)
