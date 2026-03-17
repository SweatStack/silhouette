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
        power, tte = reg.predict_inverse()
        assert len(power) == len(tte)
        assert np.all(np.diff(tte) <= 0)

    def test_get_set_params(self):
        reg = TwoParameterRegressor(method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000
