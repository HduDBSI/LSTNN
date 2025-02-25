import os

from easytorch.utils.registry import scan_modules

from .registry import SCALER_REGISTRY
from .dataset import TimeSeriesForecastingDataset
from .dataset import TimeSeriesForecastingDataset_ZhengZhou

__all__ = ["SCALER_REGISTRY", "TimeSeriesForecastingDataset", "TimeSeriesForecastingDataset_ZhengZhou"]

# fix bugs on Windows systems and on jupyter
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
scan_modules(project_dir, __file__, ["__init__.py", "registry.py"])
