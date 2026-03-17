import numpy as np
import pytest

from silhouette import OmniDurationRegressor


class TestOmniDurationRegressor:
    def test_fit_predict_roundtrip(self, power_duration_data):
        X, y = power_duration_data
        reg = OmniDurationRegressor()
        reg.fit(X, y)

        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.05)

    def test_fitted_params_are_reasonable(self, power_duration_data):
        X, y = power_duration_data
        reg = OmniDurationRegressor().fit(X, y)

        assert 200 < reg.cp_ < 300
        assert 800 < reg.p_max_ < 1400
        assert 10_000 < reg.w_prime_ < 30_000
        assert reg.a_ > 0
        assert 1200 <= reg.tcp_max_ <= 7200

    def test_predict_before_fit_raises(self):
        reg = OmniDurationRegressor()
        with pytest.raises(Exception):
            reg.predict(np.array([[60]]))

    def test_monotonically_decreasing_predictions(self, power_duration_data):
        X, y = power_duration_data
        reg = OmniDurationRegressor().fit(X, y)

        durations = np.arange(1, 3601).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)

    def test_predict_inverse(self, power_duration_data):
        X, y = power_duration_data
        reg = OmniDurationRegressor().fit(X, y)

        power, tte = reg.predict_inverse()
        assert len(power) == len(tte)
        # Higher power should have shorter time to exhaustion
        assert np.all(np.diff(tte) <= 0)

    def test_opt_result_stored(self, power_duration_data):
        X, y = power_duration_data
        reg = OmniDurationRegressor().fit(X, y)
        assert hasattr(reg, "opt_result_")
        assert reg.opt_result_.success or reg.opt_result_.fun < 100

    def test_custom_bounds(self, power_duration_data):
        X, y = power_duration_data
        reg = OmniDurationRegressor(bounds={"cp": (200, 400)})
        reg.fit(X, y)
        assert 200 <= reg.cp_ <= 400

    def test_get_set_params(self):
        reg = OmniDurationRegressor(method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000

        reg.set_params(method="Nelder-Mead")
        assert reg.get_params()["method"] == "Nelder-Mead"
