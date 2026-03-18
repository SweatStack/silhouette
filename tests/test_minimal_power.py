import numpy as np
import pytest

from silhouette import MinimalPowerPowerRegressor, MinimalPowerSpeedRegressor


@pytest.fixture
def minimal_power_data():
    """Synthetic data from the minimal power model: map=400, map_duration=300, gamma_l=0.06, gamma_s=0.1."""
    params = {"map": 400, "map_duration": 300, "gamma_l": 0.06, "gamma_s": 0.1}
    durations = np.array([60, 120, 300, 600, 1200, 2700])
    power = MinimalPowerPowerRegressor.curve(durations, **params)
    return durations.reshape(-1, 1), power


@pytest.fixture
def minimal_speed_data():
    """Synthetic data from the minimal power speed model: map=5, map_duration=300, gamma_l=0.06, gamma_s=0.1."""
    params = {"map": 5, "map_duration": 300, "gamma_l": 0.06, "gamma_s": 0.1}
    durations = np.array([60, 120, 300, 600, 1200, 2700])
    speed = MinimalPowerSpeedRegressor.curve(durations, **params)
    return durations.reshape(-1, 1), speed


class TestMinimalPowerPowerRegressor:
    def test_fit_predict_roundtrip(self, minimal_power_data):
        X, y = minimal_power_data
        reg = MinimalPowerPowerRegressor().fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.05)

    def test_fitted_params_are_reasonable(self, minimal_power_data):
        X, y = minimal_power_data
        reg = MinimalPowerPowerRegressor().fit(X, y)
        assert 300 < reg.map_ < 500
        assert 180 < reg.map_duration_ < 720
        assert 0.01 < reg.gamma_l_ < 1.0
        assert 0.01 < reg.gamma_s_ < 1.0

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            MinimalPowerPowerRegressor().predict(np.array([[60]]))

    def test_power_decreases_with_duration(self, minimal_power_data):
        X, y = minimal_power_data
        reg = MinimalPowerPowerRegressor().fit(X, y)
        durations = np.array([30, 60, 120, 300, 600, 1200]).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_get_set_params(self):
        reg = MinimalPowerPowerRegressor(max_iter=5000)
        params = reg.get_params()
        assert params["max_iter"] == 5000


class TestMinimalPowerCurve:
    def test_known_params(self):
        t = np.array([60, 300, 1200])
        power = MinimalPowerPowerRegressor.curve(
            t, map=400, map_duration=300, gamma_l=0.06, gamma_s=0.1,
        )
        assert power.shape == (3,)
        assert np.all(np.diff(power) <= 0)

    def test_scalar_input(self):
        power = MinimalPowerPowerRegressor.curve(
            300, map=400, map_duration=300, gamma_l=0.06, gamma_s=0.1,
        )
        assert isinstance(power, float)

    def test_matches_fitted_predict(self, minimal_power_data):
        X, y = minimal_power_data
        reg = MinimalPowerPowerRegressor().fit(X, y)
        from_curve = MinimalPowerPowerRegressor.curve(
            X[:, 0], map=reg.map_, map_duration=reg.map_duration_,
            gamma_l=reg.gamma_l_, gamma_s=reg.gamma_s_,
        )
        from_predict = reg.predict(X)
        np.testing.assert_allclose(from_curve, from_predict, rtol=0.01)


class TestMinimalPowerCurveInverse:
    def test_roundtrip_with_curve(self):
        params = {"map": 400, "map_duration": 300, "gamma_l": 0.06, "gamma_s": 0.1}
        t_in = np.array([60, 300, 1200])
        power = MinimalPowerPowerRegressor.curve(t_in, **params)
        t_out = MinimalPowerPowerRegressor.curve_inverse(power, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=0.05)

    def test_scalar_input(self):
        tte = MinimalPowerPowerRegressor.curve_inverse(
            400, map=400, map_duration=300, gamma_l=0.06, gamma_s=0.1,
        )
        assert isinstance(tte, float)


class TestMinimalPowerSpeedRegressor:
    def test_fit_predict_roundtrip(self, minimal_speed_data):
        X, y = minimal_speed_data
        reg = MinimalPowerSpeedRegressor().fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.05)

    def test_fitted_params_are_reasonable(self, minimal_speed_data):
        X, y = minimal_speed_data
        reg = MinimalPowerSpeedRegressor().fit(X, y)
        assert 3 < reg.map_ < 7
        assert 180 < reg.map_duration_ < 720
        assert 0.01 < reg.gamma_l_ < 1.0
        assert 0.01 < reg.gamma_s_ < 1.0

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            MinimalPowerSpeedRegressor().predict(np.array([[60]]))

    def test_speed_decreases_with_duration(self, minimal_speed_data):
        X, y = minimal_speed_data
        reg = MinimalPowerSpeedRegressor().fit(X, y)
        durations = np.array([30, 60, 120, 300, 600, 1200]).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_get_set_params(self):
        reg = MinimalPowerSpeedRegressor(max_iter=5000)
        params = reg.get_params()
        assert params["max_iter"] == 5000


class TestSpeedCurve:
    def test_known_params(self):
        t = np.array([60, 300, 1200])
        speed = MinimalPowerSpeedRegressor.curve(
            t, map=5, map_duration=300, gamma_l=0.06, gamma_s=0.1,
        )
        assert speed.shape == (3,)
        assert np.all(np.diff(speed) <= 0)

    def test_roundtrip_curve_inverse(self):
        params = {"map": 5, "map_duration": 300, "gamma_l": 0.06, "gamma_s": 0.1}
        t_in = np.array([60, 300, 1200])
        speed = MinimalPowerSpeedRegressor.curve(t_in, **params)
        t_out = MinimalPowerSpeedRegressor.curve_inverse(speed, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=0.05)
