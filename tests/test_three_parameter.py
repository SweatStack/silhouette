import numpy as np
import pytest

from silhouette import ThreeParameterRegressor


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
        power, tte = reg.predict_inverse()
        assert len(power) == len(tte)
        assert np.all(np.diff(tte) <= 0)

    def test_get_set_params(self):
        reg = ThreeParameterRegressor(method="L-BFGS-B", max_iter=5000)
        params = reg.get_params()
        assert params["method"] == "L-BFGS-B"
        assert params["max_iter"] == 5000
