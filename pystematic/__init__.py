__version__ = '0.1.0'

from .cli import pytorch_experiment, global_entrypoint, parameter_decorator as parameter

experiment = pytorch_experiment # temp alias

from .torchutil import Looper, BetterDataLoader

from .recording import Recorder

from .counter import Counter

from .pytorch_api import global_api_obj as _torchapi, PystematicPytorchAPI

import wrapt

# Import all attributes of the api object to the module level
for name in dir(PystematicPytorchAPI):
    if not name.startswith("_"):
        globals()[name] = getattr(_torchapi, name)
