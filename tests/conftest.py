import numpy as np
import pytest


@pytest.fixture
def two_param_data():
    """Synthetic data from the two-parameter model: cp=250, w_prime=20000."""
    cp, w_prime = 250, 20_000
    durations = np.array([120, 180, 300, 600, 900])
    power = cp + w_prime / durations
    return durations.reshape(-1, 1), power


@pytest.fixture
def three_param_data():
    """Synthetic data from the three-parameter model: cp=250, w_prime=20000, p_max=1100."""
    cp, w_prime, p_max = 250, 20_000, 1100
    durations = np.array([5, 10, 30, 60, 120, 300, 600, 900])
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


@pytest.fixture
def two_param_speed_data():
    """Synthetic data from the two-parameter critical speed model: cs=4, d_prime=200."""
    cs, d_prime = 4, 200
    durations = np.array([120, 180, 300, 600, 900])
    speed = cs + d_prime / durations
    return durations.reshape(-1, 1), speed


@pytest.fixture
def three_param_speed_data():
    """Synthetic data from the three-parameter critical speed model: cs=4, d_prime=300, s_max=10."""
    cs, d_prime, s_max = 4, 300, 10
    durations = np.array([5, 10, 30, 60, 120, 300, 600, 900])
    numerator = d_prime * s_max + durations * cs * (s_max - cs)
    denominator = d_prime + durations * (s_max - cs)
    speed = numerator / denominator
    return durations.reshape(-1, 1), speed


@pytest.fixture
def omni_speed_data():
    """Synthetic data from the omni-domain speed model: cs=4, s_max=10, d_prime=300, a=0.5, tcp_max=1800."""
    cs, s_max, d_prime, a, tcp_max = 4, 10, 300, 0.5, 1800
    durations = np.array([5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600])
    base = d_prime / durations * (1 - np.exp(-durations * (s_max - cs) / d_prime)) + cs
    speed = np.where(durations <= tcp_max, base, base - a * np.log(durations / tcp_max))
    return durations.reshape(-1, 1), speed
