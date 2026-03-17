import numpy as np
from scipy.optimize import minimize
from sklearn.utils.validation import check_array, check_is_fitted

from silhouette._base import BaseRegressor

DEFAULT_BOUNDS = {
    "cp": (1, 800),
    "p_max": (1, 4000),
    "w_prime": (1, 60_000),
    "a": (0, None),
    "tcp_max": (1200, 7200),
}

DEFAULT_INITIAL_PARAMS = {
    "cp": 300,
    "p_max": 1000,
    "w_prime": 20_000,
    "a": 50,
    "tcp_max": 1800,
}


class OmniDurationRegressor(BaseRegressor):
    """Omni-domain power-duration model.

    Fits the model from Puchowicz et al. (2020) to cycling power data:

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

    def __init__(self, bounds=None, initial_params=None, method="Nelder-Mead", max_iter=10_000):
        self.bounds = bounds
        self.initial_params = initial_params
        self.method = method
        self.max_iter = max_iter

    def _resolve_bounds(self):
        merged = dict(DEFAULT_BOUNDS)
        if self.bounds is not None:
            merged.update(self.bounds)
        return [merged[k] for k in self._PARAM_ORDER]

    def _resolve_initial_params(self):
        merged = dict(DEFAULT_INITIAL_PARAMS)
        if self.initial_params is not None:
            merged.update(self.initial_params)
        return [merged[k] for k in self._PARAM_ORDER]

    @staticmethod
    def _omni_model(t, cp, p_max, w_prime, a, tcp_max):
        base = w_prime / t * (1 - np.exp(-t * (p_max - cp) / w_prime)) + cp
        return np.where(t <= tcp_max, base, base - a * np.log(t / tcp_max))

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
            prediction = self._omni_model(X[:, 0], *params)
            return self._cost_function(prediction, y)

        result = minimize(
            objective,
            self._resolve_initial_params(),
            method=self.method,
            bounds=self._resolve_bounds(),
            options={"maxiter": self.max_iter},
        )

        self.cp_, self.p_max_, self.w_prime_, self.a_, self.tcp_max_ = result.x
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
        return self._omni_model(X[:, 0], self.cp_, self.p_max_, self.w_prime_, self.a_, self.tcp_max_)

    def predict_inverse(self, max_duration=7200):
        """Predict time to exhaustion for a range of power outputs.

        Parameters
        ----------
        max_duration : int, default=7200
            Maximum duration in seconds to consider.

        Returns
        -------
        power : ndarray
            Power values from 1 W up to (but not including) p_max.
        tte : ndarray
            Corresponding time-to-exhaustion in seconds.
        """
        check_is_fitted(self)
        t = np.arange(1, max_duration + 1)
        predicted_power = self.predict(t.reshape(-1, 1))

        power = np.arange(1, int(self.p_max_))
        tte = np.interp(power, predicted_power[::-1], t[::-1])

        return power, tte

    def _more_tags(self):
        return {"poor_score": True}
