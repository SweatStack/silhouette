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
