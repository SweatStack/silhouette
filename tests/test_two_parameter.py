import warnings

import numpy as np
import pytest

from silhouette import TwoParameterRegressor


class TestCurve:
    def test_known_params(self):
        t = np.array([120, 300, 600])
        power = TwoParameterRegressor.curve(t, cp=250, w_prime=20_000)
        expected = 250 + 20_000 / t
        np.testing.assert_allclose(power, expected)

    def test_scalar_input(self):
        power = TwoParameterRegressor.curve(300, cp=250, w_prime=20_000)
        assert float(power) == pytest.approx(250 + 20_000 / 300)

    def test_matches_fitted_predict(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor().fit(X, y)
        from_curve = TwoParameterRegressor.curve(X[:, 0], cp=reg.cp_, w_prime=reg.w_prime_)
        from_predict = reg.predict(X)
        np.testing.assert_allclose(from_curve, from_predict)


class TestCurveInverse:
    def test_known_params(self):
        tte = TwoParameterRegressor.curve_inverse(
            np.array([300, 350, 400]), cp=250, w_prime=20_000,
        )
        expected = 20_000 / (np.array([300, 350, 400]) - 250)
        np.testing.assert_allclose(tte, expected, rtol=1e-6)

    def test_scalar_input(self):
        tte = TwoParameterRegressor.curve_inverse(350, cp=250, w_prime=20_000)
        assert isinstance(tte, float)

    def test_roundtrip_with_curve(self):
        params = {"cp": 250, "w_prime": 20_000}
        t_in = np.array([120, 300, 600])
        power = TwoParameterRegressor.curve(t_in, **params)
        t_out = TwoParameterRegressor.curve_inverse(power, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=1e-6)


class TestTwoParameterRegressor:
    def test_fit_predict_roundtrip(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor().fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.01)

    def test_fitted_params_are_reasonable(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor().fit(X, y)
        assert 200 < reg.cp_ < 300
        assert 15_000 < reg.w_prime_ < 25_000

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            TwoParameterRegressor().predict(np.array([[60]]))

    def test_power_decreases_with_duration(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor().fit(X, y)
        durations = np.arange(60, 1201).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_approaches_cp_at_long_durations(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor().fit(X, y)
        long_duration = np.array([[100_000]])
        assert abs(reg.predict(long_duration)[0] - reg.cp_) < 1

    def test_predict_inverse(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor().fit(X, y)
        power = np.array([300, 350, 400])
        tte = reg.predict_inverse(power)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_predict_inverse_roundtrip(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor().fit(X, y)
        power_in = np.array([300, 350, 400])
        tte = reg.predict_inverse(power_in)
        power_out = reg.predict(tte.reshape(-1, 1))
        np.testing.assert_allclose(power_out, power_in, rtol=1e-6)

    def test_get_set_params(self):
        reg = TwoParameterRegressor(method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000


class TestFittingParameter:
    def test_nonlinear_same_as_default(self, two_param_data):
        X, y = two_param_data
        reg_default = TwoParameterRegressor().fit(X, y)
        reg_nonlinear = TwoParameterRegressor(fitting="nonlinear").fit(X, y)
        assert reg_default.cp_ == pytest.approx(reg_nonlinear.cp_)
        assert reg_default.w_prime_ == pytest.approx(reg_nonlinear.w_prime_)

    def test_work_duration_produces_valid_params(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor(fitting="work_duration").fit(X, y)
        assert 200 < reg.cp_ < 300
        assert 15_000 < reg.w_prime_ < 25_000

    def test_work_duration_differs_from_nonlinear(self):
        """With noisy data, work_duration and nonlinear produce different estimates."""
        np.random.seed(42)
        cp_true, w_prime_true = 250, 20_000
        durations = np.array([120, 180, 300, 600, 1200])
        power = cp_true + w_prime_true / durations + np.random.normal(0, 5, len(durations))
        X = durations.reshape(-1, 1)

        reg_nl = TwoParameterRegressor(fitting="nonlinear").fit(X, power)
        reg_wd = TwoParameterRegressor(fitting="work_duration").fit(X, power)

        # They should not be exactly equal with noisy data
        assert reg_nl.cp_ != pytest.approx(reg_wd.cp_, abs=0.01)

    def test_work_duration_with_bounds_warns(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor(fitting="work_duration", bounds={"cp": (1, 500)})
        with pytest.warns(UserWarning, match="bounds is ignored"):
            reg.fit(X, y)

    def test_work_duration_with_initial_params_warns(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor(
            fitting="work_duration", initial_params={"cp": 200}
        )
        with pytest.warns(UserWarning, match="initial_params is ignored"):
            reg.fit(X, y)

    def test_invalid_fitting_raises(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor(fitting="invalid")
        with pytest.raises(ValueError, match="Invalid fitting"):
            reg.fit(X, y)

    def test_predict_after_work_duration(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor(fitting="work_duration").fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape
        # Predictions should be reasonable (close to actual for clean data)
        np.testing.assert_allclose(y_pred, y, rtol=0.01)

    def test_opt_result_is_none_for_work_duration(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor(fitting="work_duration").fit(X, y)
        assert reg.opt_result_ is None


class TestDurationRange:
    def test_filters_data(self):
        """Only data within the range should be used for fitting."""
        cp, w_prime = 250, 20_000
        durations = np.array([30, 120, 300, 600, 1200, 3600])
        power = cp + w_prime / durations
        X = durations.reshape(-1, 1)

        reg = TwoParameterRegressor(duration_range=(120, 900)).fit(X, power)
        # 30s and 1200s and 3600s are outside range
        assert reg.duration_mask_.sum() == 3  # 120, 300, 600
        assert 200 < reg.cp_ < 300

    def test_predict_works_outside_range(self):
        """predict() should work at any duration, even outside the fitted range."""
        cp, w_prime = 250, 20_000
        durations = np.array([120, 300, 600])
        power = cp + w_prime / durations
        X = durations.reshape(-1, 1)

        reg = TwoParameterRegressor(duration_range=(120, 900)).fit(X, power)
        # Predict at 5s and 3600s — outside the fitting range
        y_pred = reg.predict(np.array([[5], [3600]]))
        assert y_pred.shape == (2,)
        assert y_pred[0] > y_pred[1]  # shorter duration = more power

    def test_duration_mask_attribute(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor(duration_range=(120, 900)).fit(X, y)
        assert hasattr(reg, 'duration_mask_')
        assert reg.duration_mask_.dtype == bool
        assert len(reg.duration_mask_) == len(X)

    def test_mask_all_true_when_no_range(self, two_param_data):
        X, y = two_param_data
        reg = TwoParameterRegressor().fit(X, y)
        assert reg.duration_mask_.all()

    def test_one_sided_lower(self):
        cp, w_prime = 250, 20_000
        durations = np.array([30, 60, 120, 300, 600])
        power = cp + w_prime / durations
        X = durations.reshape(-1, 1)

        reg = TwoParameterRegressor(duration_range=(120, None)).fit(X, power)
        assert reg.duration_mask_.sum() == 3  # 120, 300, 600

    def test_one_sided_upper(self):
        cp, w_prime = 250, 20_000
        durations = np.array([120, 300, 600, 1200, 3600])
        power = cp + w_prime / durations
        X = durations.reshape(-1, 1)

        reg = TwoParameterRegressor(duration_range=(None, 900)).fit(X, power)
        assert reg.duration_mask_.sum() == 3  # 120, 300, 600

    def test_too_few_samples_after_filter(self):
        durations = np.array([30, 60, 3600, 7200])
        power = np.array([500, 400, 260, 255])
        X = durations.reshape(-1, 1)

        reg = TwoParameterRegressor(duration_range=(120, 900))
        with pytest.raises(ValueError, match="duration_range.*filters"):
            reg.fit(X, power)

    def test_warns_outside_recommended_range(self):
        """Should warn when data is outside the recommended range and duration_range is not set."""
        cp, w_prime = 250, 20_000
        durations = np.array([5, 120, 300, 600, 3600])  # 5s and 3600s are outside
        power = cp + w_prime / durations
        X = durations.reshape(-1, 1)

        reg = TwoParameterRegressor()
        with pytest.warns(UserWarning, match="designed for durations between"):
            reg.fit(X, power)

    def test_no_warning_when_data_in_range(self, two_param_data):
        """Should not warn when all data is within the recommended range."""
        X, y = two_param_data  # durations: 120, 180, 300, 600, 1200
        # 1200 > 900 so this will actually warn. Use a subset.
        mask = X[:, 0] <= 900
        reg = TwoParameterRegressor()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reg.fit(X[mask], y[mask])

    def test_no_warning_when_duration_range_set(self):
        """Should not warn when duration_range is explicitly set, even if data is outside."""
        cp, w_prime = 250, 20_000
        durations = np.array([5, 120, 300, 600, 3600])
        power = cp + w_prime / durations
        X = durations.reshape(-1, 1)

        reg = TwoParameterRegressor(duration_range=(120, 900))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reg.fit(X, power)

    def test_works_with_work_duration_fitting(self):
        cp, w_prime = 250, 20_000
        durations = np.array([30, 120, 300, 600, 3600])
        power = cp + w_prime / durations
        X = durations.reshape(-1, 1)

        reg = TwoParameterRegressor(
            fitting="work_duration", duration_range=(120, 900)
        ).fit(X, power)
        assert reg.duration_mask_.sum() == 3
        assert 200 < reg.cp_ < 300

    def test_gridsearchable(self):
        """duration_range should work with sklearn GridSearchCV."""
        from sklearn.model_selection import GridSearchCV

        cp, w_prime = 250, 20_000
        durations = np.array([60, 120, 180, 300, 600, 900, 1200])
        power = cp + w_prime / durations
        X = durations.reshape(-1, 1)

        param_grid = {"duration_range": [(120, 600), (120, 900), (60, 1200)]}
        gs = GridSearchCV(TwoParameterRegressor(), param_grid, cv=2)
        gs.fit(X, power)
        assert hasattr(gs, "best_params_")
