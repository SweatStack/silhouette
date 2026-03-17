import numpy as np
from scipy.optimize import brentq, minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BaseRegressor(RegressorMixin, BaseEstimator):
    """Base class for power-duration regressors.

    Subclasses must define:
        _PARAM_ORDER: tuple of parameter names
        _DEFAULT_BOUNDS: dict of parameter bounds
        _DEFAULT_INITIAL_PARAMS: dict of initial parameter estimates
        curve(t, **params): static method computing the power-duration curve
    """

    _PARAM_ORDER = ()
    _DEFAULT_BOUNDS = {}
    _DEFAULT_INITIAL_PARAMS = {}

    def __init__(self, bounds=None, initial_params=None, method="Nelder-Mead", max_iter=10_000):
        self.bounds = bounds
        self.initial_params = initial_params
        self.method = method
        self.max_iter = max_iter

    def _resolve_bounds(self):
        merged = dict(self._DEFAULT_BOUNDS)
        if self.bounds is not None:
            merged.update(self.bounds)
        return [merged[k] for k in self._PARAM_ORDER]

    def _resolve_initial_params(self):
        merged = dict(self._DEFAULT_INITIAL_PARAMS)
        if self.initial_params is not None:
            merged.update(self.initial_params)
        return [merged[k] for k in self._PARAM_ORDER]

    def _fitted_params(self):
        return {name: getattr(self, f"{name}_") for name in self._PARAM_ORDER}

    @staticmethod
    def curve(t, **params):
        """Evaluate the power-duration curve at given durations.

        This is the mathematical model itself. It can be called directly as a
        class method without fitting, when the model parameters are already
        known.

        Parameters
        ----------
        t : array-like
            Durations in seconds.
        **params
            Model parameters as keyword arguments (e.g. cp=250, w_prime=20000).

        Returns
        -------
        power : ndarray
            Predicted power in watts.
        """
        raise NotImplementedError

    @classmethod
    def curve_inverse(cls, y, **params):
        """Evaluate the inverse: find the duration for a given power output.

        Solves curve(t, ...) = y for t using root finding. This is the inverse
        of curve(): given a power output, it returns how long that power can be
        sustained.

        The power values must be within the range of the model. For values at
        or below critical power, the duration is theoretically infinite.

        Parameters
        ----------
        y : array-like
            Power values in watts.
        **params
            Model parameters as keyword arguments (e.g. cp=250, w_prime=20000).

        Returns
        -------
        tte : ndarray
            Time to exhaustion in seconds.
        """
        y = np.asarray(y, dtype=float)
        scalar = y.ndim == 0
        y = np.atleast_1d(y)

        def solve_one(power):
            return brentq(lambda t: cls.curve(t, **params) - power, 1, 1e7)

        tte = np.array([solve_one(p) for p in y])
        return float(tte[0]) if scalar else tte

    def _preprocess_data(self, X, y):
        """Validate, sort by duration ascending, and enforce monotonically decreasing power."""
        X, y = check_X_y(X, y, ensure_min_samples=2)

        order = np.argsort(X[:, 0])
        X = X[order]
        y = y[order]

        # Enforce monotonically decreasing: walk backwards and take cumulative max
        y = np.maximum.accumulate(y[::-1])[::-1]

        return X, y

    def _cost_function(self, prediction, y):
        return np.mean((prediction - y) ** 2)

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
            return self._cost_function(self.curve(t, **params), y)

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

    def predict(self, X):
        """Predict power for given durations.

        Uses the fitted model parameters. Equivalent to calling
        ``curve(t, **fitted_params)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Durations in seconds.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted power in watts.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.curve(X[:, 0], **self._fitted_params())

    def predict_inverse(self, y):
        """Predict time to exhaustion for given power outputs.

        Uses the fitted model parameters. Equivalent to calling
        ``curve_inverse(y, **fitted_params)``.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Power values in watts.

        Returns
        -------
        tte : ndarray of shape (n_samples,)
            Time to exhaustion in seconds.
        """
        check_is_fitted(self)
        return self.curve_inverse(y, **self._fitted_params())

    def _more_tags(self):
        return {"poor_score": True}
