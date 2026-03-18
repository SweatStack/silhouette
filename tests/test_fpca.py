import numpy as np
import pytest

from silhouette import FPCAPowerRegressor


@pytest.fixture
def reg():
    return FPCAPowerRegressor.from_model()


@pytest.fixture
def synthetic_data(reg):
    """A power-duration curve generated from known FPC scores."""
    fpc1, fpc2, fpc3 = 0.3, -0.1, 0.05
    t = reg.population_model.time_grid
    power = FPCAPowerRegressor.curve(t, fpc1=fpc1, fpc2=fpc2, fpc3=fpc3)
    return t.reshape(-1, 1), power, (fpc1, fpc2, fpc3)


class TestFromModel:
    def test_loads_bundled_model(self):
        reg = FPCAPowerRegressor.from_model()
        assert reg.population_model is not None
        assert reg.population_model.mean_function.shape == (90,)
        assert reg.population_model.eigenfunctions.shape == (90, 3)

    def test_time_grid(self, reg):
        assert reg.population_model.time_grid[0] == 1
        assert reg.population_model.time_grid[-1] == 7200
        assert len(reg.population_model.time_grid) == 90

    def test_without_model_raises(self):
        reg = FPCAPowerRegressor()
        with pytest.raises(ValueError, match="No population model"):
            reg.fit(np.array([[60]]), np.array([400]))


class TestCurve:
    def test_known_scores(self):
        t = np.array([60, 300, 1200])
        power = FPCAPowerRegressor.curve(t, fpc1=0.0, fpc2=0.0, fpc3=0.0)
        assert len(power) == 3
        assert np.all(power > 0)
        # Zero scores should give the population mean
        assert power[0] > power[1] > power[2]

    def test_scalar_input(self):
        power = FPCAPowerRegressor.curve(300, fpc1=0.0, fpc2=0.0, fpc3=0.0)
        assert isinstance(float(power), float)

    def test_higher_fpc1_means_more_power(self):
        t = np.array([300])
        low = FPCAPowerRegressor.curve(t, fpc1=-0.5, fpc2=0.0, fpc3=0.0)
        high = FPCAPowerRegressor.curve(t, fpc1=0.5, fpc2=0.0, fpc3=0.0)
        assert high[0] > low[0]


class TestCurveInverse:
    def test_roundtrip_with_curve(self):
        params = {"fpc1": 0.3, "fpc2": -0.1, "fpc3": 0.0}
        t_in = np.array([60, 300, 1200])
        power = FPCAPowerRegressor.curve(t_in, **params)
        t_out = FPCAPowerRegressor.curve_inverse(power, **params)
        np.testing.assert_allclose(t_out, t_in, rtol=1e-4)

    def test_scalar_input(self):
        tte = FPCAPowerRegressor.curve_inverse(300, fpc1=0.0, fpc2=0.0, fpc3=0.0)
        assert isinstance(tte, float)


class TestFit:
    def test_fit_recovers_scores(self, reg, synthetic_data):
        X, y, (fpc1, fpc2, fpc3) = synthetic_data
        reg.fit(X, y)
        assert reg.fpc1_ == pytest.approx(fpc1, abs=0.01)
        assert reg.fpc2_ == pytest.approx(fpc2, abs=0.01)
        assert reg.fpc3_ == pytest.approx(fpc3, abs=0.01)

    def test_fit_sets_attributes(self, reg, synthetic_data):
        X, y, _ = synthetic_data
        reg.fit(X, y)
        assert hasattr(reg, "fpc1_")
        assert hasattr(reg, "fpc2_")
        assert hasattr(reg, "fpc3_")
        assert hasattr(reg, "population_scores_")
        assert hasattr(reg, "time_grid_")

    def test_predict_before_fit_raises(self, reg):
        with pytest.raises(Exception):
            reg.predict(np.array([[60]]))

    def test_fit_with_sparse_data(self, reg):
        """Fit with fewer points than the standard grid."""
        t = np.array([30, 120, 600, 1800]).reshape(-1, 1)
        power = np.array([600, 400, 310, 260])
        reg.fit(t, power)
        assert hasattr(reg, "fpc1_")


class TestPredict:
    def test_predict_roundtrip(self, reg, synthetic_data):
        X, y, _ = synthetic_data
        reg.fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_allclose(y_pred, y, rtol=0.01)

    def test_predict_at_arbitrary_durations(self, reg, synthetic_data):
        X, y, _ = synthetic_data
        reg.fit(X, y)
        new_t = np.array([[45], [150], [900]])
        pred = reg.predict(new_t)
        assert pred.shape == (3,)
        assert np.all(pred > 0)

    def test_monotonically_decreasing(self, reg, synthetic_data):
        X, y, _ = synthetic_data
        reg.fit(X, y)
        durations = np.arange(5, 3601).reshape(-1, 1)
        predictions = reg.predict(durations)
        assert np.all(np.diff(predictions) <= 0)


class TestPredictInverse:
    def test_roundtrip(self, reg, synthetic_data):
        X, y, _ = synthetic_data
        reg.fit(X, y)
        power_in = np.array([300, 400, 500])
        tte = reg.predict_inverse(power_in)
        power_out = reg.predict(tte.reshape(-1, 1))
        np.testing.assert_allclose(power_out, power_in, rtol=1e-4)


class TestPercentiles:
    def test_returns_dict(self, reg, synthetic_data):
        X, y, _ = synthetic_data
        reg.fit(X, y)
        pct = reg.percentiles()
        assert set(pct.keys()) == {"fpc1", "fpc2", "fpc3"}
        for v in pct.values():
            assert 0 <= v <= 100

    def test_before_fit_raises(self, reg):
        with pytest.raises(Exception):
            reg.percentiles()


class TestZScores:
    def test_returns_dict(self, reg, synthetic_data):
        X, y, _ = synthetic_data
        reg.fit(X, y)
        zs = reg.z_scores()
        assert set(zs.keys()) == {"fpc1", "fpc2", "fpc3"}
        for v in zs.values():
            assert isinstance(v, float)

    def test_zero_scores_near_zero_z(self, reg):
        """FPC scores of zero should be near the population mean, so z ~ 0."""
        t = reg.population_model.time_grid.reshape(-1, 1)
        mean_power = np.exp(reg.population_model.mean_function)
        reg.fit(t, mean_power)
        zs = reg.z_scores()
        for v in zs.values():
            assert abs(v) < 0.5
