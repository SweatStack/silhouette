import numpy as np
import pytest

from silhouette import ExpPowerRegressor, ExpSpeedRegressor


@pytest.fixture
def exp_data():
    """Synthetic data from the exp model: cp=250, p_max=1100, tau=60."""
    cp, p_max, tau = 250, 1100, 60
    durations = np.array([5, 10, 30, 60, 120, 300, 600, 900])
    power = (p_max - cp) * np.exp(-durations / tau) + cp
    return durations.reshape(-1, 1), power


@pytest.fixture
def exp_speed_data():
    """Synthetic data from the exp speed model: cs=4, s_max=10, tau=60."""
    cs, s_max, tau = 4, 10, 60
    durations = np.array([5, 10, 30, 60, 120, 300, 600, 900])
    speed = (s_max - cs) * np.exp(-durations / tau) + cs
    return durations.reshape(-1, 1), speed


class TestCurve:
    def test_known_params(self):
        t = np.array([5, 60, 600])
        power = ExpPowerRegressor.curve(t, cp=250, p_max=1100, tau=60)
        expected = (1100 - 250) * np.exp(-t / 60) + 250
        np.testing.assert_allclose(power, expected)

    def test_scalar_input(self):
        power = ExpPowerRegressor.curve(300, cp=250, p_max=1100, tau=60)
        assert isinstance(float(power), float)

    def test_matches_fitted_predict(self, exp_data):
        X, y = exp_data
        reg = ExpPowerRegressor().fit(X, y)
        from_curve = ExpPowerRegressor.curve(
            X[:, 0], cp=reg.cp_, p_max=reg.p_max_, tau=reg.tau_,
        )
        from_predict = reg.predict(X)
        np.testing.assert_allclose(from_curve, from_predict)


class TestCurveInverse:
    def test_known_params(self):
        params = {"cp": 250, "p_max": 1100, "tau": 60}
        tte = ExpPowerRegressor.curve_inverse(np.array([300, 400, 500]), **params)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_scalar_input(self):
        tte = ExpPowerRegressor.curve_inverse(400, cp=250, p_max=1100, tau=60)
        assert isinstance(tte, float)

    def test_roundtrip_with_curve(self):
        params = {"cp": 250, "p_max": 1100, "tau": 60}
        t_in = np.array([5, 60, 600])
        power = ExpPowerRegressor.curve(t_in, **params)
        t_out = ExpPowerRegressor.curve_inverse(power, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=1e-6)


class TestExpPowerRegressor:
    def test_fit_predict_roundtrip(self, exp_data):
        X, y = exp_data
        reg = ExpPowerRegressor().fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.01)

    def test_fitted_params_are_reasonable(self, exp_data):
        X, y = exp_data
        reg = ExpPowerRegressor().fit(X, y)
        assert 200 < reg.cp_ < 300
        assert 800 < reg.p_max_ < 1400
        assert 30 < reg.tau_ < 120

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            ExpPowerRegressor().predict(np.array([[60]]))

    def test_power_decreases_with_duration(self, exp_data):
        X, y = exp_data
        reg = ExpPowerRegressor().fit(X, y)
        durations = np.arange(1, 1201).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_bounded_by_p_max_and_cp(self, exp_data):
        X, y = exp_data
        reg = ExpPowerRegressor().fit(X, y)
        durations = np.arange(1, 3601).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(predictions <= reg.p_max_ + 1)
        assert np.all(predictions >= reg.cp_ - 1)

    def test_predict_inverse(self, exp_data):
        X, y = exp_data
        reg = ExpPowerRegressor().fit(X, y)
        power = np.array([300, 400, 500])
        tte = reg.predict_inverse(power)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_predict_inverse_roundtrip(self, exp_data):
        X, y = exp_data
        reg = ExpPowerRegressor().fit(X, y)
        power_in = np.array([300, 400, 500])
        tte = reg.predict_inverse(power_in)
        power_out = reg.predict(tte.reshape(-1, 1))
        np.testing.assert_allclose(power_out, power_in, rtol=1e-6)

    def test_get_set_params(self):
        reg = ExpPowerRegressor(method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000


class TestExpSpeedRegressor:
    def test_fit_predict_roundtrip(self, exp_speed_data):
        X, y = exp_speed_data
        reg = ExpSpeedRegressor().fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.01)

    def test_fitted_params_are_reasonable(self, exp_speed_data):
        X, y = exp_speed_data
        reg = ExpSpeedRegressor().fit(X, y)
        assert 3 < reg.cs_ < 5
        assert 8 < reg.s_max_ < 12
        assert 30 < reg.tau_ < 120

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            ExpSpeedRegressor().predict(np.array([[60]]))

    def test_speed_decreases_with_duration(self, exp_speed_data):
        X, y = exp_speed_data
        reg = ExpSpeedRegressor().fit(X, y)
        durations = np.arange(1, 901).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_bounded_by_s_max_and_cs(self, exp_speed_data):
        X, y = exp_speed_data
        reg = ExpSpeedRegressor().fit(X, y)
        durations = np.arange(1, 901).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(predictions <= reg.s_max_ + 0.1)
        assert np.all(predictions >= reg.cs_ - 0.1)

    def test_predict_inverse(self, exp_speed_data):
        X, y = exp_speed_data
        reg = ExpSpeedRegressor().fit(X, y)
        speed = np.array([5.0, 6.0, 7.0])
        tte = reg.predict_inverse(speed)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_predict_inverse_roundtrip(self, exp_speed_data):
        X, y = exp_speed_data
        reg = ExpSpeedRegressor().fit(X, y)
        speed_in = np.array([5.0, 6.0, 7.0])
        tte = reg.predict_inverse(speed_in)
        speed_out = reg.predict(tte.reshape(-1, 1))
        np.testing.assert_allclose(speed_out, speed_in, rtol=1e-6)

    def test_get_set_params(self):
        reg = ExpSpeedRegressor(method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000


class TestSpeedCurve:
    def test_known_params(self):
        t = np.array([5, 60, 600])
        speed = ExpSpeedRegressor.curve(t, cs=4, s_max=10, tau=60)
        expected = (10 - 4) * np.exp(-t / 60) + 4
        np.testing.assert_allclose(speed, expected)

    def test_scalar_input(self):
        speed = ExpSpeedRegressor.curve(300, cs=4, s_max=10, tau=60)
        assert isinstance(float(speed), float)

    def test_matches_fitted_predict(self, exp_speed_data):
        X, y = exp_speed_data
        reg = ExpSpeedRegressor().fit(X, y)
        from_curve = ExpSpeedRegressor.curve(
            X[:, 0], cs=reg.cs_, s_max=reg.s_max_, tau=reg.tau_,
        )
        from_predict = reg.predict(X)
        np.testing.assert_allclose(from_curve, from_predict)


class TestSpeedCurveInverse:
    def test_known_params(self):
        params = {"cs": 4, "s_max": 10, "tau": 60}
        tte = ExpSpeedRegressor.curve_inverse(np.array([5.0, 6.0, 7.0]), **params)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_roundtrip_with_curve(self):
        params = {"cs": 4, "s_max": 10, "tau": 60}
        t_in = np.array([5, 60, 600])
        speed = ExpSpeedRegressor.curve(t_in, **params)
        t_out = ExpSpeedRegressor.curve_inverse(speed, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=1e-6)
