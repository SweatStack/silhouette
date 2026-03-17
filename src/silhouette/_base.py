import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y


class BaseRegressor(RegressorMixin, BaseEstimator):
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
