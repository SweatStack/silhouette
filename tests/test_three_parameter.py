import numpy as np
import pytest

from silhouette import (
    ThreeParamCriticalPowerRegressor,
    ThreeParamCriticalSpeedRegressor,
)


class TestCurve:
    def test_known_params(self):
        t = np.array([5, 60, 600])
        power = ThreeParamCriticalPowerRegressor.curve(t, cp=250, w_prime=20_000, p_max=1100)
        num = 20_000 * 1100 + t * 250 * (1100 - 250)
        den = 20_000 + t * (1100 - 250)
        np.testing.assert_allclose(power, num / den)

    def test_scalar_input(self):
        power = ThreeParamCriticalPowerRegressor.curve(300, cp=250, w_prime=20_000, p_max=1100)
        assert isinstance(float(power), float)

    def test_matches_fitted_predict(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParamCriticalPowerRegressor().fit(X, y)
        from_curve = ThreeParamCriticalPowerRegressor.curve(
            X[:, 0], cp=reg.cp_, w_prime=reg.w_prime_, p_max=reg.p_max_,
        )
        from_predict = reg.predict(X)
        np.testing.assert_allclose(from_curve, from_predict)


class TestCurveInverse:
    def test_known_params(self):
        params = {"cp": 250, "w_prime": 20_000, "p_max": 1100}
        tte = ThreeParamCriticalPowerRegressor.curve_inverse(np.array([300, 400, 500]), **params)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_scalar_input(self):
        tte = ThreeParamCriticalPowerRegressor.curve_inverse(
            400, cp=250, w_prime=20_000, p_max=1100,
        )
        assert isinstance(tte, float)

    def test_roundtrip_with_curve(self):
        params = {"cp": 250, "w_prime": 20_000, "p_max": 1100}
        t_in = np.array([5, 60, 600])
        power = ThreeParamCriticalPowerRegressor.curve(t_in, **params)
        t_out = ThreeParamCriticalPowerRegressor.curve_inverse(power, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=1e-6)


class TestThreeParamCriticalPowerRegressor:
    def test_fit_predict_roundtrip(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParamCriticalPowerRegressor().fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.01)

    def test_fitted_params_are_reasonable(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParamCriticalPowerRegressor().fit(X, y)
        assert 200 < reg.cp_ < 300
        assert 15_000 < reg.w_prime_ < 25_000
        assert 800 < reg.p_max_ < 1400

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            ThreeParamCriticalPowerRegressor().predict(np.array([[60]]))

    def test_power_decreases_with_duration(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParamCriticalPowerRegressor().fit(X, y)
        durations = np.arange(1, 1201).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_bounded_by_p_max_and_cp(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParamCriticalPowerRegressor().fit(X, y)
        durations = np.arange(1, 3601).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(predictions <= reg.p_max_ + 1)
        assert np.all(predictions >= reg.cp_ - 1)

    def test_predict_inverse(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParamCriticalPowerRegressor().fit(X, y)
        power = np.array([300, 400, 500])
        tte = reg.predict_inverse(power)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_predict_inverse_roundtrip(self, three_param_data):
        X, y = three_param_data
        reg = ThreeParamCriticalPowerRegressor().fit(X, y)
        power_in = np.array([300, 400, 500])
        tte = reg.predict_inverse(power_in)
        power_out = reg.predict(tte.reshape(-1, 1))
        np.testing.assert_allclose(power_out, power_in, rtol=1e-6)

    def test_get_set_params(self):
        reg = ThreeParamCriticalPowerRegressor(method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000


class TestThreeParamCriticalSpeedRegressor:
    def test_fit_predict_roundtrip(self, three_param_speed_data):
        X, y = three_param_speed_data
        reg = ThreeParamCriticalSpeedRegressor().fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.01)

    def test_fitted_params_are_reasonable(self, three_param_speed_data):
        X, y = three_param_speed_data
        reg = ThreeParamCriticalSpeedRegressor().fit(X, y)
        assert 3 < reg.cs_ < 5
        assert 200 < reg.d_prime_ < 400
        assert 8 < reg.s_max_ < 12

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            ThreeParamCriticalSpeedRegressor().predict(np.array([[60]]))

    def test_speed_decreases_with_duration(self, three_param_speed_data):
        X, y = three_param_speed_data
        reg = ThreeParamCriticalSpeedRegressor().fit(X, y)
        durations = np.arange(1, 901).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_bounded_by_s_max_and_cs(self, three_param_speed_data):
        X, y = three_param_speed_data
        reg = ThreeParamCriticalSpeedRegressor().fit(X, y)
        durations = np.arange(1, 901).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(predictions <= reg.s_max_ + 0.1)
        assert np.all(predictions >= reg.cs_ - 0.1)

    def test_predict_inverse(self, three_param_speed_data):
        X, y = three_param_speed_data
        reg = ThreeParamCriticalSpeedRegressor().fit(X, y)
        speed = np.array([5.0, 6.0, 7.0])
        tte = reg.predict_inverse(speed)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_predict_inverse_roundtrip(self, three_param_speed_data):
        X, y = three_param_speed_data
        reg = ThreeParamCriticalSpeedRegressor().fit(X, y)
        speed_in = np.array([5.0, 6.0, 7.0])
        tte = reg.predict_inverse(speed_in)
        speed_out = reg.predict(tte.reshape(-1, 1))
        np.testing.assert_allclose(speed_out, speed_in, rtol=1e-6)

    def test_get_set_params(self):
        reg = ThreeParamCriticalSpeedRegressor(method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000


class TestSpeedCurve:
    def test_known_params(self):
        t = np.array([5, 60, 600])
        speed = ThreeParamCriticalSpeedRegressor.curve(t, cs=4, d_prime=300, s_max=10)
        num = 300 * 10 + t * 4 * (10 - 4)
        den = 300 + t * (10 - 4)
        np.testing.assert_allclose(speed, num / den)

    def test_scalar_input(self):
        speed = ThreeParamCriticalSpeedRegressor.curve(300, cs=4, d_prime=300, s_max=10)
        assert isinstance(float(speed), float)

    def test_matches_fitted_predict(self, three_param_speed_data):
        X, y = three_param_speed_data
        reg = ThreeParamCriticalSpeedRegressor().fit(X, y)
        from_curve = ThreeParamCriticalSpeedRegressor.curve(
            X[:, 0], cs=reg.cs_, d_prime=reg.d_prime_, s_max=reg.s_max_,
        )
        from_predict = reg.predict(X)
        np.testing.assert_allclose(from_curve, from_predict)


class TestSpeedCurveInverse:
    def test_known_params(self):
        params = {"cs": 4, "d_prime": 300, "s_max": 10}
        tte = ThreeParamCriticalSpeedRegressor.curve_inverse(
            np.array([5.0, 6.0, 7.0]), **params,
        )
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_roundtrip_with_curve(self):
        params = {"cs": 4, "d_prime": 300, "s_max": 10}
        t_in = np.array([5, 60, 600])
        speed = ThreeParamCriticalSpeedRegressor.curve(t_in, **params)
        t_out = ThreeParamCriticalSpeedRegressor.curve_inverse(speed, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=1e-6)
