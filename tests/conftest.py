import numpy as np
import pytest


@pytest.fixture
def two_param_data():
    """Synthetic data from the two-parameter model: cp=250, w_prime=20000."""
    cp, w_prime = 250, 20_000
    durations = np.array([120, 180, 300, 600, 1200])
    power = cp + w_prime / durations
    return durations.reshape(-1, 1), power


@pytest.fixture
def three_param_data():
    """Synthetic data from the three-parameter model: cp=250, w_prime=20000, p_max=1100."""
    cp, w_prime, p_max = 250, 20_000, 1100
    durations = np.array([5, 10, 30, 60, 120, 300, 600, 1200])
    numerator = w_prime * p_max + durations * cp * (p_max - cp)
    denominator = w_prime + durations * (p_max - cp)
    power = numerator / denominator
    return durations.reshape(-1, 1), power


@pytest.fixture
def omni_data():
    """Synthetic data from the omni model: cp=250, p_max=1100, w_prime=20000, a=40, tcp_max=1800."""
    cp, p_max, w_prime, a, tcp_max = 250, 1100, 20_000, 40, 1800
    durations = np.array([5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600])
    base = w_prime / durations * (1 - np.exp(-durations * (p_max - cp) / w_prime)) + cp
    power = np.where(durations <= tcp_max, base, base - a * np.log(durations / tcp_max))
    return durations.reshape(-1, 1), power
