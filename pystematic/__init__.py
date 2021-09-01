import importlib_metadata as _importlib_metadata

__version__ = _importlib_metadata.version(__name__)

from . import core as _core

parameter = _core.parameter_decorator
experiment = _core.experiment_decorator
group = _core.group_decorator
param_group = _core.parameter_group_decorator

_core.app.load_all_plugins()
