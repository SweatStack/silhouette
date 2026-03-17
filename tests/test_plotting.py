import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from silhouette import TwoParameterRegressor, OmniDurationRegressor, FPCARegressor
from silhouette.plotting import PowerDurationDisplay, ModeOfVarianceDisplay


@pytest.fixture
def data():
    durations = np.array([30, 60, 120, 300, 600, 1200])
    power = np.array([600, 480, 400, 340, 310, 290])
    return durations.reshape(-1, 1), power


@pytest.fixture
def fitted_2p(data):
    X, y = data
    return TwoParameterRegressor().fit(X, y)


@pytest.fixture
def fitted_omni(data):
    X, y = data
    return OmniDurationRegressor().fit(X, y)


@pytest.fixture
def fitted_fpca(data):
    X, y = data
    reg = FPCARegressor.from_model()
    return reg.fit(X, y)


class TestPowerDurationDisplay:
    def test_from_estimator(self, fitted_omni, data):
        X, y = data
        display = PowerDurationDisplay.from_estimator(fitted_omni, X, y)
        assert display.ax_ is not None
        assert display.figure_ is not None
        assert display.scatter_ is not None
        assert len(display.lines_) == 1

    def test_from_estimator_without_data(self, fitted_omni):
        display = PowerDurationDisplay.from_estimator(fitted_omni)
        assert display.scatter_ is None
        assert len(display.lines_) == 1

    def test_from_estimator_custom_name(self, fitted_omni, data):
        X, y = data
        display = PowerDurationDisplay.from_estimator(
            fitted_omni, X, y, name="My model",
        )
        assert display.lines_[0].get_label() == "My model"

    def test_from_estimator_default_name(self, fitted_omni, data):
        X, y = data
        display = PowerDurationDisplay.from_estimator(fitted_omni, X, y)
        assert display.lines_[0].get_label() == "OmniDurationRegressor"

    def test_from_estimators(self, fitted_2p, fitted_omni, data):
        X, y = data
        display = PowerDurationDisplay.from_estimators(
            [fitted_2p, fitted_omni], X, y,
        )
        assert len(display.lines_) == 2
        assert display.scatter_ is not None

    def test_from_estimators_custom_names(self, fitted_2p, fitted_omni, data):
        X, y = data
        display = PowerDurationDisplay.from_estimators(
            [fitted_2p, fitted_omni], X, y,
            names=["2P", "Omni"],
        )
        assert display.lines_[0].get_label() == "2P"
        assert display.lines_[1].get_label() == "Omni"

    def test_from_estimators_default_names(self, fitted_2p, fitted_omni, data):
        X, y = data
        display = PowerDurationDisplay.from_estimators(
            [fitted_2p, fitted_omni], X, y,
        )
        assert display.lines_[0].get_label() == "TwoParameterRegressor"
        assert display.lines_[1].get_label() == "OmniDurationRegressor"

    def test_custom_ax(self, fitted_omni, data):
        import matplotlib.pyplot as plt
        X, y = data
        fig, ax = plt.subplots()
        display = PowerDurationDisplay.from_estimator(fitted_omni, X, y, ax=ax)
        assert display.ax_ is ax

    def test_fpca_regressor(self, fitted_fpca, data):
        X, y = data
        display = PowerDurationDisplay.from_estimator(fitted_fpca, X, y)
        assert len(display.lines_) == 1


class TestModeOfVarianceDisplay:
    def test_from_model_all_components(self):
        display = ModeOfVarianceDisplay.from_model()
        assert isinstance(display.axes_, np.ndarray)
        assert len(display.axes_) == 3
        assert display.athlete_line_ is None

    def test_from_model_single_component(self):
        display = ModeOfVarianceDisplay.from_model(component=2)
        assert not isinstance(display.axes_, np.ndarray)
        assert display.athlete_line_ is None

    def test_from_estimator(self, fitted_fpca):
        display = ModeOfVarianceDisplay.from_estimator(fitted_fpca)
        assert isinstance(display.axes_, np.ndarray)
        assert len(display.axes_) == 3
        assert display.athlete_line_ is not None

    def test_from_estimator_single_component(self, fitted_fpca):
        display = ModeOfVarianceDisplay.from_estimator(fitted_fpca, component=1)
        assert not isinstance(display.axes_, np.ndarray)
        assert display.athlete_line_ is not None

    def test_custom_n_sd(self):
        display = ModeOfVarianceDisplay.from_model(n_sd=1, n_lines=10)
        assert display.figure_ is not None
