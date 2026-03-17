import numpy as np
import pytest

from silhouette import ThreeParameterRegressor


class TestCurve:
    def test_known_params(self):
        t = np.array([5, 60, 600])
        power = ThreeParameterRegressor.curve(t, cp=250, w_prime=20_000, p_max=1100)
        num = 20_000 * 1100 + t * 250 * (1100 - 250)
        den = 20_000 + t * (1100 - 250)
        np.testing.assert_allclose(power, num / den)

    def test_scalar_input(self):
        power = ThreeParameterRegressor.curve(300, cp=250, w_prime=20_000, p_max=1100)
        assert isinstance(float(power), float)

    def test_matches_fitted_predict(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParameterRegressor().fit(X, y)
        from_curve = ThreeParameterRegressor.curve(
            X[:, 0], cp=reg.cp_, w_prime=reg.w_prime_, p_max=reg.p_max_,
        )
        from_predict = reg.predict(X)
        np.testing.assert_allclose(from_curve, from_predict)


class TestCurveInverse:
    def test_known_params(self):
        params = {"cp": 250, "w_prime": 20_000, "p_max": 1100}
        tte = ThreeParameterRegressor.curve_inverse(np.array([300, 400, 500]), **params)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_scalar_input(self):
        tte = ThreeParameterRegressor.curve_inverse(
            400, cp=250, w_prime=20_000, p_max=1100,
        )
        assert isinstance(tte, float)

    def test_roundtrip_with_curve(self):
        params = {"cp": 250, "w_prime": 20_000, "p_max": 1100}
        t_in = np.array([5, 60, 600])
        power = ThreeParameterRegressor.curve(t_in, **params)
        t_out = ThreeParameterRegressor.curve_inverse(power, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=1e-6)


class TestThreeParameterRegressor:
    def test_fit_predict_roundtrip(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParameterRegressor().fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.01)

    def test_fitted_params_are_reasonable(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParameterRegressor().fit(X, y)
        assert 200 < reg.cp_ < 300
        assert 15_000 < reg.w_prime_ < 25_000
        assert 800 < reg.p_max_ < 1400

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            ThreeParameterRegressor().predict(np.array([[60]]))

    def test_power_decreases_with_duration(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParameterRegressor().fit(X, y)
        durations = np.arange(1, 1201).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_bounded_by_p_max_and_cp(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParameterRegressor().fit(X, y)
        durations = np.arange(1, 3601).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(predictions <= reg.p_max_ + 1)
        assert np.all(predictions >= reg.cp_ - 1)

    def test_predict_inverse(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParameterRegressor().fit(X, y)
        power = np.array([300, 400, 500])
        tte = reg.predict_inverse(power)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_predict_inverse_roundtrip(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParameterRegressor().fit(X, y)
        power_in = np.array([300, 400, 500])
        tte = reg.predict_inverse(power_in)
        power_out = reg.predict(tte.reshape(-1, 1))
        np.testing.assert_allclose(power_out, power_in, rtol=1e-6)

    def test_get_set_params(self):
        reg = ThreeParameterRegressor(method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000
