from importlib.metadata import version

from silhouette.two_parameter import (
    TwoParamCriticalPowerRegressor,
    TwoParamCriticalSpeedRegressor,
)
from silhouette.three_parameter import (
    ThreeParamCriticalPowerRegressor,
    ThreeParamCriticalSpeedRegressor,
)
from silhouette.omni import (
    OmniDomainPowerRegressor,
    OmniDomainSpeedRegressor,
)
from silhouette.minimal_power import (
    MinimalPowerPowerRegressor,
    MinimalPowerSpeedRegressor,
)
from silhouette.fpca import FPCAPowerRegressor

__version__ = version("silhouette")

__all__ = [
    "TwoParamCriticalPowerRegressor",
    "TwoParamCriticalSpeedRegressor",
    "ThreeParamCriticalPowerRegressor",
    "ThreeParamCriticalSpeedRegressor",
    "OmniDomainPowerRegressor",
    "OmniDomainSpeedRegressor",
    "MinimalPowerPowerRegressor",
    "MinimalPowerSpeedRegressor",
    "FPCAPowerRegressor",
]
