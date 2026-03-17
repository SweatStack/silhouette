import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BaseRegressor(RegressorMixin, BaseEstimator):
    """Base class for power-duration regressors.

    Subclasses must define:
        _PARAM_ORDER: tuple of parameter names
        _DEFAULT_BOUNDS: dict of parameter bounds
        _DEFAULT_INITIAL_PARAMS: dict of initial parameter estimates
        _model(t, ...): static method computing the power-duration curve
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

    @staticmethod
    def _model(t, *params):
        raise NotImplementedError

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

        def objective(params):
            prediction = self._model(X[:, 0], *params)
            return self._cost_function(prediction, y)

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
        params = [getattr(self, f"{name}_") for name in self._PARAM_ORDER]
        return self._model(X[:, 0], *params)

    def predict_inverse(self, max_duration=7200):
        """Predict time to exhaustion for a range of power outputs.

        Generates a dense forward prediction, then interpolates to find the
        duration at each integer watt from 1 W up to the predicted power at t=1s.

        Parameters
        ----------
        max_duration : int, default=7200
            Maximum duration in seconds to consider.

        Returns
        -------
        power : ndarray
            Power values in watts.
        tte : ndarray
            Corresponding time-to-exhaustion in seconds.
        """
        check_is_fitted(self)
        t = np.arange(1, max_duration + 1)
        predicted_power = self.predict(t.reshape(-1, 1))

        max_power = int(predicted_power[0])
        power = np.arange(1, max_power)
        tte = np.interp(power, predicted_power[::-1], t[::-1])

        return power, tte

    def _more_tags(self):
        return {"poor_score": True}
