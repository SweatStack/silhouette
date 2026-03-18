import numpy as np
import pytest

from silhouette import VDOTPowerRegressor, VDOTSpeedRegressor
from silhouette.vdot import (
    _vo2_from_speed, _speed_from_vo2,
    _vo2_from_power, _power_from_vo2,
    _fraction_vo2max,
)


@pytest.fixture
def vdot_power_data():
    """Synthetic data from the VDOT power model: vdot=50, body_mass=75."""
    durations = np.array([180, 300, 600, 900, 1200, 1800, 3600, 7200])
    power = VDOTPowerRegressor.curve(durations, vdot=50, body_mass=75)
    return durations.reshape(-1, 1), power


@pytest.fixture
def vdot_data():
    """Synthetic data from the VDOT model: vdot=50."""
    durations = np.array([180, 300, 600, 900, 1200, 1800, 3600, 7200])
    speed = VDOTSpeedRegressor.curve(durations, vdot=50)
    return durations.reshape(-1, 1), speed


class TestVDOTFormulas:
    def test_vo2_speed_roundtrip(self):
        speeds = np.array([3.0, 4.0, 5.0, 6.0])
        vo2 = _vo2_from_speed(speeds)
        recovered = _speed_from_vo2(vo2)
        np.testing.assert_allclose(recovered, speeds, rtol=1e-10)

    def test_fraction_asymptotic_behavior(self):
        """f(t) should approach 0.8 as t -> infinity."""
        f_long = _fraction_vo2max(1e6)
        assert abs(f_long - 0.8) < 1e-3

    def test_fraction_at_zero(self):
        """f(0) = 0.8 + 0.1894393 + 0.2989558 ≈ 1.2884."""
        f0 = _fraction_vo2max(0)
        np.testing.assert_allclose(f0, 1.2883951, rtol=1e-5)

    def test_fraction_decreases_with_duration(self):
        durations = np.array([60, 300, 600, 1800, 3600, 7200])
        fractions = _fraction_vo2max(durations)
        assert np.all(np.diff(fractions) < 0)

    def test_vo2_power_roundtrip(self):
        powers = np.array([100, 200, 300, 400])
        body_mass = 75.0
        vo2 = _vo2_from_power(powers, body_mass)
        recovered = _power_from_vo2(vo2, body_mass)
        np.testing.assert_allclose(recovered, powers, rtol=1e-10)

    def test_vo2_power_known_value(self):
        """200W at 70kg: VO2 = 200 * 11.7 / 70 ≈ 33.43 ml/kg/min."""
        vo2 = _vo2_from_power(200, 70)
        np.testing.assert_allclose(vo2, 200 * 11.7 / 70, rtol=1e-10)


class TestCurve:
    def test_known_values(self):
        """VDOT=50 at t=1800s should give ~4.10 m/s (~6:28/mi pace)."""
        speed = VDOTSpeedRegressor.curve(1800, vdot=50)
        assert 4.0 < float(speed) < 4.2

    def test_scalar_input(self):
        speed = VDOTSpeedRegressor.curve(600, vdot=50)
        assert isinstance(float(speed), float)

    def test_matches_fitted_predict(self, vdot_data):
        X, y = vdot_data
        reg = VDOTSpeedRegressor().fit(X, y)
        from_curve = VDOTSpeedRegressor.curve(X[:, 0], vdot=reg.vdot_)
        from_predict = reg.predict(X)
        np.testing.assert_allclose(from_curve, from_predict)


class TestCurveInverse:
    def test_roundtrip_with_curve(self):
        params = {"vdot": 50}
        t_in = np.array([300, 600, 1800])
        speed = VDOTSpeedRegressor.curve(t_in, **params)
        t_out = VDOTSpeedRegressor.curve_inverse(speed, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=1e-4)

    def test_scalar_input(self):
        tte = VDOTSpeedRegressor.curve_inverse(4.0, vdot=50)
        assert isinstance(tte, float)


class TestVDOTSpeedRegressor:
    def test_fit_predict_roundtrip(self, vdot_data):
        X, y = vdot_data
        reg = VDOTSpeedRegressor().fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.01)

    def test_fitted_params_are_reasonable(self, vdot_data):
        X, y = vdot_data
        reg = VDOTSpeedRegressor().fit(X, y)
        assert 45 < reg.vdot_ < 55

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            VDOTSpeedRegressor().predict(np.array([[600]]))

    def test_speed_decreases_with_duration(self, vdot_data):
        X, y = vdot_data
        reg = VDOTSpeedRegressor().fit(X, y)
        durations = np.arange(180, 7201).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_predict_inverse(self, vdot_data):
        X, y = vdot_data
        reg = VDOTSpeedRegressor().fit(X, y)
        speed = np.array([4.0, 4.5, 5.0])
        tte = reg.predict_inverse(speed)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_predict_inverse_roundtrip(self, vdot_data):
        X, y = vdot_data
        reg = VDOTSpeedRegressor().fit(X, y)
        speed_in = np.array([4.0, 4.5, 5.0])
        tte = reg.predict_inverse(speed_in)
        speed_out = reg.predict(tte.reshape(-1, 1))
        np.testing.assert_allclose(speed_out, speed_in, rtol=1e-4)

    def test_get_set_params(self):
        reg = VDOTSpeedRegressor(method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000

    def test_sklearn_clone(self, vdot_data):
        from sklearn.base import clone

        X, y = vdot_data
        reg = VDOTSpeedRegressor().fit(X, y)
        reg2 = clone(reg)
        reg2.fit(X, y)
        np.testing.assert_allclose(reg.vdot_, reg2.vdot_, rtol=0.01)


class TestDurationRange:
    def test_warns_outside_recommended_range(self):
        durations = np.array([30, 60, 120, 300, 600])
        speed = VDOTSpeedRegressor.curve(durations, vdot=50)
        with pytest.warns(UserWarning, match="designed for durations"):
            VDOTSpeedRegressor().fit(durations.reshape(-1, 1), speed)

    def test_duration_range_filters(self):
        durations = np.array([60, 180, 300, 600, 900, 1800, 3600, 10800])
        speed = VDOTSpeedRegressor.curve(durations, vdot=50)
        reg = VDOTSpeedRegressor(duration_range=(180, 7200))
        reg.fit(durations.reshape(-1, 1), speed)
        assert reg.duration_mask_.sum() == 6  # 180, 300, 600, 900, 1800, 3600

    def test_predicts_outside_fitted_range(self, vdot_data):
        X, y = vdot_data
        reg = VDOTSpeedRegressor().fit(X, y)
        # Should still predict at durations outside training range
        pred = reg.predict(np.array([[60], [10800]]))
        assert len(pred) == 2
        assert pred[0] > pred[1]


class TestPowerCurve:
    def test_scalar_input(self):
        power = VDOTPowerRegressor.curve(600, vdot=50, body_mass=75)
        assert isinstance(float(power), float)

    def test_matches_fitted_predict(self, vdot_power_data):
        X, y = vdot_power_data
        reg = VDOTPowerRegressor(body_mass=75).fit(X, y)
        from_curve = VDOTPowerRegressor.curve(X[:, 0], vdot=reg.vdot_, body_mass=75)
        from_predict = reg.predict(X)
        np.testing.assert_allclose(from_curve, from_predict)

    def test_higher_body_mass_more_power(self):
        """At same VDOT, heavier athlete produces more absolute watts."""
        p_light = VDOTPowerRegressor.curve(600, vdot=50, body_mass=60)
        p_heavy = VDOTPowerRegressor.curve(600, vdot=50, body_mass=90)
        assert float(p_heavy) > float(p_light)


class TestPowerCurveInverse:
    def test_roundtrip_with_curve(self):
        params = {"vdot": 50, "body_mass": 75}
        t_in = np.array([300, 600, 1800])
        power = VDOTPowerRegressor.curve(t_in, **params)
        t_out = VDOTPowerRegressor.curve_inverse(power, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=1e-4)

    def test_scalar_input(self):
        tte = VDOTPowerRegressor.curve_inverse(300, vdot=50, body_mass=75)
        assert isinstance(tte, float)


class TestVDOTPowerRegressor:
    def test_fit_predict_roundtrip(self, vdot_power_data):
        X, y = vdot_power_data
        reg = VDOTPowerRegressor(body_mass=75).fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.01)

    def test_fitted_params_are_reasonable(self, vdot_power_data):
        X, y = vdot_power_data
        reg = VDOTPowerRegressor(body_mass=75).fit(X, y)
        assert 45 < reg.vdot_ < 55

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            VDOTPowerRegressor(body_mass=75).predict(np.array([[600]]))

    def test_power_decreases_with_duration(self, vdot_power_data):
        X, y = vdot_power_data
        reg = VDOTPowerRegressor(body_mass=75).fit(X, y)
        durations = np.arange(180, 7201).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_predict_inverse(self, vdot_power_data):
        X, y = vdot_power_data
        reg = VDOTPowerRegressor(body_mass=75).fit(X, y)
        power = np.array([280, 300, 350])
        tte = reg.predict_inverse(power)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_predict_inverse_roundtrip(self, vdot_power_data):
        X, y = vdot_power_data
        reg = VDOTPowerRegressor(body_mass=75).fit(X, y)
        power_in = np.array([280, 300, 350])
        tte = reg.predict_inverse(power_in)
        power_out = reg.predict(tte.reshape(-1, 1))
        np.testing.assert_allclose(power_out, power_in, rtol=1e-4)

    def test_get_set_params(self):
        reg = VDOTPowerRegressor(body_mass=80, method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000
        assert params["body_mass"] == 80

    def test_sklearn_clone(self, vdot_power_data):
        from sklearn.base import clone

        X, y = vdot_power_data
        reg = VDOTPowerRegressor(body_mass=75).fit(X, y)
        reg2 = clone(reg)
        assert reg2.body_mass == 75
        reg2.fit(X, y)
        np.testing.assert_allclose(reg.vdot_, reg2.vdot_, rtol=0.01)

    def test_body_mass_affects_vdot(self):
        """Same power data with different body_mass should give different VDOT."""
        durations = np.array([180, 300, 600, 900, 1800, 3600])
        power = np.array([350, 320, 290, 275, 260, 240])
        reg1 = VDOTPowerRegressor(body_mass=65).fit(durations.reshape(-1, 1), power)
        reg2 = VDOTPowerRegressor(body_mass=85).fit(durations.reshape(-1, 1), power)
        assert reg1.vdot_ != reg2.vdot_
