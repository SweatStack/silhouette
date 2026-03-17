from importlib import resources

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


class _PopulationModel:
    """Container for the pre-fitted FPCA population model."""

    __slots__ = (
        "mean_function",
        "eigenfunctions",
        "eigenvalues",
        "explained_variance_ratio",
        "pc_scores",
        "time_grid",
    )

    def __init__(self, path):
        data = np.load(path)
        self.mean_function = data["mean_function"]
        self.eigenfunctions = data["eigenfunctions"]
        self.eigenvalues = data["eigenvalues"]
        self.explained_variance_ratio = data["explained_variance_ratio"]
        self.pc_scores = data["pc_scores"]
        self.time_grid = data["time_grid"]

    @property
    def n_components(self):
        return self.eigenfunctions.shape[1]

    @property
    def log_time_grid(self):
        return np.log(self.time_grid)


def _load_bundled_model():
    ref = resources.files("silhouette").joinpath("_fpca_model.npz")
    with resources.as_file(ref) as path:
        return _PopulationModel(path)


# Lazy-loaded class-level cache for curve/curve_inverse
_cached_model = None


def _get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = _load_bundled_model()
    return _cached_model


def _interpolate_to_grid(durations, power, time_grid):
    """Interpolate power data to the standard time grid using log-linear interpolation."""
    f = interp1d(
        np.log(durations),
        np.log(power),
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return np.exp(f(np.log(time_grid)))


def _evaluate_curve(t, fpc1, fpc2, fpc3, model):
    """Evaluate the F3 model at given durations."""
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    log_t = np.log(t)

    # Interpolate the mean function and eigenfunctions to the requested durations
    log_grid = model.log_time_grid
    mean_at_t = np.interp(log_t, log_grid, model.mean_function)

    scores = np.array([fpc1, fpc2, fpc3])
    eigen_at_t = np.column_stack(
        [np.interp(log_t, log_grid, model.eigenfunctions[:, i]) for i in range(3)]
    )

    result = np.exp(mean_at_t + eigen_at_t @ scores)
    return float(result[0]) if scalar else result


class FPCARegressor(RegressorMixin, BaseEstimator):
    """FPCA power-duration model (F3 model).

    A data-driven power-duration model based on Functional Principal Component
    Analysis. The model structure (mean function and eigenfunctions) is learned
    from a population of 2000 cyclists. Individual athletes are characterized
    by three FPC scores:

    - fpc1: Overall power level (gain). Explains ~77% of population variance.
    - fpc2: Sprint vs endurance bias. Explains ~16% of population variance.
    - fpc3: Mid-duration specialization. Explains ~2% of population variance.

    The model must be initialized with a population model via ``from_model()``.

    Parameters
    ----------
    population_model : _PopulationModel
        Pre-fitted population model. Use ``FPCARegressor.from_model()`` to
        create an instance.

    Attributes
    ----------
    fpc1_ : float
        First FPC score (overall power level).
    fpc2_ : float
        Second FPC score (sprint vs endurance bias).
    fpc3_ : float
        Third FPC score (mid-duration specialization).
    population_scores_ : ndarray of shape (n_training, 3)
        FPC scores from the training population, for percentile/z-score
        calculations.
    time_grid_ : ndarray of shape (90,)
        Standard time grid in seconds.

    References
    ----------
    Puchowicz, M. J., & Skiba, P. F. (2025). Functional Data Analysis of the
    Power-Duration Relationship in Cyclists. International Journal of Sports
    Physiology and Performance, 1(aop), 1-10.
    """

    _PARAM_ORDER = ("fpc1", "fpc2", "fpc3")

    def __init__(self, population_model=None):
        self.population_model = population_model

    @classmethod
    def from_model(cls, path=None):
        """Create an FPCARegressor with a pre-fitted population model.

        Parameters
        ----------
        path : str or Path, optional
            Path to a custom population model (.npz file). If None, loads
            the bundled model that ships with the library.

        Returns
        -------
        FPCARegressor
            Ready to fit to individual athlete data.
        """
        if path is None:
            model = _load_bundled_model()
        else:
            model = _PopulationModel(path)
        return cls(population_model=model)

    def _check_population_model(self):
        if self.population_model is None:
            raise ValueError(
                "No population model loaded. Use FPCARegressor.from_model() "
                "to create an instance."
            )

    def fit(self, X, y):
        """Fit the model to an athlete's duration-power data.

        Projects the athlete's power-duration curve onto the population
        eigenfunctions to obtain FPC scores. The athlete's data is
        interpolated to the standard 90-point time grid if needed.

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
        self._check_population_model()
        X, y = check_X_y(X, y, ensure_min_samples=2)
        model = self.population_model

        # Interpolate to the standard time grid
        aligned = _interpolate_to_grid(X[:, 0], y, model.time_grid)

        # Project onto eigenfunctions
        log_aligned = np.log(aligned)
        centered = log_aligned - model.mean_function
        scores = centered @ model.eigenfunctions

        self.fpc1_, self.fpc2_, self.fpc3_ = scores[:3]
        self.population_scores_ = model.pc_scores
        self.time_grid_ = model.time_grid
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
        return _evaluate_curve(
            X[:, 0], self.fpc1_, self.fpc2_, self.fpc3_, self.population_model,
        )

    def predict_inverse(self, y):
        """Predict time to exhaustion for given power outputs.

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
        return self.curve_inverse(
            y, fpc1=self.fpc1_, fpc2=self.fpc2_, fpc3=self.fpc3_,
        )

    def percentiles(self):
        """Percentile ranking of fitted scores against the training population.

        Returns
        -------
        dict
            Percentile (0-100) for each FPC score.
        """
        check_is_fitted(self)
        scores = np.array([self.fpc1_, self.fpc2_, self.fpc3_])
        return {
            name: float((score > self.population_scores_[:, i]).mean() * 100)
            for i, (name, score) in enumerate(zip(self._PARAM_ORDER, scores))
        }

    def z_scores(self):
        """Z-scores of fitted parameters relative to the training population.

        Returns
        -------
        dict
            Z-score for each FPC score.
        """
        check_is_fitted(self)
        scores = np.array([self.fpc1_, self.fpc2_, self.fpc3_])
        means = self.population_scores_.mean(axis=0)
        stds = self.population_scores_.std(axis=0)
        return {
            name: float((score - means[i]) / stds[i])
            for i, (name, score) in enumerate(zip(self._PARAM_ORDER, scores))
        }

    @classmethod
    def curve(cls, t, *, fpc1, fpc2, fpc3):
        """Evaluate the F3 power-duration curve at given durations.

        Can be called without fitting, when the FPC scores are already known.
        The bundled population model is lazy-loaded on first call.

        Parameters
        ----------
        t : array-like
            Durations in seconds.
        fpc1 : float
            First FPC score (overall power level).
        fpc2 : float
            Second FPC score (sprint vs endurance bias).
        fpc3 : float
            Third FPC score (mid-duration specialization).

        Returns
        -------
        power : ndarray
            Predicted power in watts.
        """
        return _evaluate_curve(t, fpc1, fpc2, fpc3, _get_model())

    @classmethod
    def curve_inverse(cls, y, *, fpc1, fpc2, fpc3):
        """Find the duration for a given power output.

        Solves curve(t, ...) = y for t using root finding.

        Parameters
        ----------
        y : array-like
            Power values in watts.
        fpc1, fpc2, fpc3 : float
            FPC scores.

        Returns
        -------
        tte : float or ndarray
            Time to exhaustion in seconds.
        """
        model = _get_model()
        y = np.asarray(y, dtype=float)
        scalar = y.ndim == 0
        y = np.atleast_1d(y)

        def solve_one(power):
            return brentq(
                lambda t: _evaluate_curve(t, fpc1, fpc2, fpc3, model) - power,
                1, 1e7,
            )

        tte = np.array([solve_one(p) for p in y])
        return float(tte[0]) if scalar else tte

    def _more_tags(self):
        return {"poor_score": True}
