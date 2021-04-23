__version__ = '0.1.0'


# from .click_adapter import global_entrypoint, parameter_decorator as parameter, label_decorator as label

# from .cli import pytorch_experiment

# experiment = pytorch_experiment # temp alias

from .core import (
    experiment_decorator as experiment, 
    experiment_group_decorator as group,
    parameter_decorator as parameter
)

from .torchutil import BetterDataLoader

from .recording import Recorder

from .counter import Counter

from .torchapi import *
