__version__ = '0.2.0'


from . import pluginapi as _pluginapi
from . import core as _core

_pluginapi.load_all_plugins()

_api_object = _pluginapi.construct_api_extension()

for name in dir(_api_object):
    if not name.startswith("_"):
        globals()[name] = getattr(_api_object, name)

parameter = _core.parameter_decorator
experiment = _core.experiment_decorator
group = _core.group_decorator