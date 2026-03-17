import numpy as np
import pytest


@pytest.fixture
def power_duration_data():
    """Synthetic power-duration curve generated from known omni model parameters.

    Parameters: cp=250, p_max=1100, w_prime=20000, a=40, tcp_max=1800
    """
    cp, p_max, w_prime, a, tcp_max = 250, 1100, 20000, 40, 1800
    durations = np.array([5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600])

    base = w_prime / durations * (1 - np.exp(-durations * (p_max - cp) / w_prime)) + cp
    power = np.where(durations <= tcp_max, base, base - a * np.log(durations / tcp_max))

    return durations.reshape(-1, 1), power
