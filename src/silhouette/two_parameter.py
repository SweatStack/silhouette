import warnings

import numpy as np

from silhouette._base import BaseRegressor


class TwoParameterRegressor(BaseRegressor):
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
        t = np.asarray(t)
        return cp + w_prime / t
