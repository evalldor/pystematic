__version__ = '0.9.0'

from . import core as _core

parameter = _core.parameter_decorator
experiment = _core.experiment_decorator
group = _core.group_decorator

_core.app.load_all_plugins()
