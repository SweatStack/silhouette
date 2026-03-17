import numpy as np
import pytest

from silhouette import OmniDurationRegressor


class TestCurve:
    def test_known_params(self):
        t = np.array([60, 1800, 3600])
        power = OmniDurationRegressor.curve(
            t, cp=250, p_max=1100, w_prime=20_000, a=40, tcp_max=1800,
        )
        assert len(power) == 3
        assert power[0] > power[1] > power[2]

    def test_scalar_input(self):
        power = OmniDurationRegressor.curve(
            300, cp=250, p_max=1100, w_prime=20_000, a=40, tcp_max=1800,
        )
        assert isinstance(float(power), float)

    def test_matches_fitted_predict(self, omni_data):
        X, y = omni_data
        reg = OmniDurationRegressor().fit(X, y)
        from_curve = OmniDurationRegressor.curve(
            X[:, 0],
            cp=reg.cp_, p_max=reg.p_max_, w_prime=reg.w_prime_,
            a=reg.a_, tcp_max=reg.tcp_max_,
        )
        from_predict = reg.predict(X)
        np.testing.assert_allclose(from_curve, from_predict)


class TestCurveInverse:
    def test_known_params(self):
        params = {"cp": 250, "p_max": 1100, "w_prime": 20_000, "a": 40, "tcp_max": 1800}
        tte = OmniDurationRegressor.curve_inverse(np.array([280, 350, 500]), **params)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_scalar_input(self):
        tte = OmniDurationRegressor.curve_inverse(
            350, cp=250, p_max=1100, w_prime=20_000, a=40, tcp_max=1800,
        )
        assert isinstance(tte, float)

    def test_roundtrip_with_curve(self):
        params = {"cp": 250, "p_max": 1100, "w_prime": 20_000, "a": 40, "tcp_max": 1800}
        t_in = np.array([10, 300, 3600])
        power = OmniDurationRegressor.curve(t_in, **params)
        t_out = OmniDurationRegressor.curve_inverse(power, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=1e-6)


class TestOmniDurationRegressor:
    def test_fit_predict_roundtrip(self, omni_data):
        X, y = omni_data
        reg = OmniDurationRegressor().fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.05)

    def test_fitted_params_are_reasonable(self, omni_data):
        X, y = omni_data
        reg = OmniDurationRegressor().fit(X, y)
        assert 200 < reg.cp_ < 300
        assert 800 < reg.p_max_ < 1400
        assert 10_000 < reg.w_prime_ < 30_000
        assert reg.a_ > 0
        assert 1200 <= reg.tcp_max_ <= 7200

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            OmniDurationRegressor().predict(np.array([[60]]))

    def test_monotonically_decreasing_predictions(self, omni_data):
        X, y = omni_data
        reg = OmniDurationRegressor().fit(X, y)
        durations = np.arange(1, 3601).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_predict_inverse(self, omni_data):
        X, y = omni_data
        reg = OmniDurationRegressor().fit(X, y)
        power = np.array([280, 350, 500])
        tte = reg.predict_inverse(power)
        assert tte.shape == (3,)
        assert np.all(np.diff(tte) <= 0)

    def test_predict_inverse_roundtrip(self, omni_data):
        X, y = omni_data
        reg = OmniDurationRegressor().fit(X, y)
        power_in = np.array([280, 350, 500])
        tte = reg.predict_inverse(power_in)
        power_out = reg.predict(tte.reshape(-1, 1))
        np.testing.assert_allclose(power_out, power_in, rtol=1e-6)

    def test_opt_result_stored(self, omni_data):
        X, y = omni_data
        reg = OmniDurationRegressor().fit(X, y)
        assert hasattr(reg, "opt_result_")
        assert reg.opt_result_.success or reg.opt_result_.fun < 100

    def test_custom_bounds(self, omni_data):
        X, y = omni_data
        reg = OmniDurationRegressor(bounds={"cp": (200, 400)}).fit(X, y)
        assert 200 <= reg.cp_ <= 400

    def test_get_set_params(self):
        reg = OmniDurationRegressor(method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000
