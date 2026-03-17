from importlib.metadata import version

from silhouette.two_parameter import TwoParameterRegressor
from silhouette.three_parameter import ThreeParameterRegressor
from silhouette.omni import OmniDurationRegressor

__version__ = version("silhouette")

__all__ = [
    "TwoParameterRegressor",
    "ThreeParameterRegressor",
    "OmniDurationRegressor",
]
