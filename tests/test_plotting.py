import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from silhouette import TwoParamCriticalPowerRegressor, OmniDomainPowerRegressor, FPCAPowerRegressor  # noqa: E501
from silhouette import MinimalPowerPowerRegressor, MinimalPowerSpeedRegressor
from silhouette.plotting import PowerDurationDisplay, ModeOfVarianceDisplay, MinimalPowerDisplay


@pytest.fixture
def data():
    durations = np.array([120, 180, 300, 600, 900])
    power = np.array([400, 370, 340, 310, 290])
    return durations.reshape(-1, 1), power


@pytest.fixture
def fitted_2p(data):
    X, y = data
    return TwoParamCriticalPowerRegressor().fit(X, y)


@pytest.fixture
def fitted_omni(data):
    X, y = data
    return OmniDomainPowerRegressor().fit(X, y)


@pytest.fixture
def fitted_fpca(data):
    X, y = data
    reg = FPCAPowerRegressor.from_model()
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
        assert display.lines_[0].get_label() == "OmniDomainPowerRegressor"

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
        assert display.lines_[0].get_label() == "TwoParamCriticalPowerRegressor"
        assert display.lines_[1].get_label() == "OmniDomainPowerRegressor"

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


class TestMinimalPowerDisplay:
    @pytest.fixture
    def fitted_power(self):
        params = {"map": 400, "map_duration": 300, "gamma_l": 0.06, "gamma_s": 0.1}
        durations = np.array([60, 120, 300, 600, 1200, 2700])
        power = MinimalPowerPowerRegressor.curve(durations, **params)
        X = durations.reshape(-1, 1)
        return MinimalPowerPowerRegressor().fit(X, power), X, power

    @pytest.fixture
    def fitted_speed(self):
        params = {"map": 5, "map_duration": 300, "gamma_l": 0.06, "gamma_s": 0.1}
        durations = np.array([60, 120, 300, 600, 1200, 2700])
        speed = MinimalPowerSpeedRegressor.curve(durations, **params)
        X = durations.reshape(-1, 1)
        return MinimalPowerSpeedRegressor().fit(X, speed), X, speed

    def test_from_estimator_power(self, fitted_power):
        reg, X, y = fitted_power
        display = MinimalPowerDisplay.from_estimator(reg, X, y)
        assert display.ax_ is not None
        assert display.figure_ is not None
        assert display.line_ is not None
        assert display.scatter_ is not None
        assert display.band_ is not None

    def test_from_estimator_speed(self, fitted_speed):
        reg, X, y = fitted_speed
        display = MinimalPowerDisplay.from_estimator(reg, X, y)
        assert display.ax_ is not None
        assert display.scatter_ is not None

    def test_without_data(self, fitted_power):
        reg, _, _ = fitted_power
        display = MinimalPowerDisplay.from_estimator(reg)
        assert display.scatter_ is None
        assert display.line_ is not None

    def test_without_reference_band(self, fitted_power):
        reg, X, y = fitted_power
        display = MinimalPowerDisplay.from_estimator(reg, X, y, reference_band=False)
        assert display.band_ is None

    def test_custom_name(self, fitted_power):
        reg, X, y = fitted_power
        display = MinimalPowerDisplay.from_estimator(reg, X, y, name="My model")
        assert display.line_.get_label() == "My model"

    def test_power_axes_labels(self, fitted_power):
        reg, X, y = fitted_power
        display = MinimalPowerDisplay.from_estimator(reg, X, y)
        assert "W" in display.ax_.get_xlabel()
        assert "MAP" in display.ax_.get_ylabel()

    def test_speed_axes_labels(self, fitted_speed):
        reg, X, y = fitted_speed
        display = MinimalPowerDisplay.from_estimator(reg, X, y)
        assert "d" in display.ax_.get_xlabel()
        assert "v" in display.ax_.get_ylabel()
