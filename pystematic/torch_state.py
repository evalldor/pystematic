import typing
import dataclasses

class TorchExperiment:

    @property
    def params(self):
        pass

    @property
    def context(self):
        pass

    @property
    def output_dir(self):
        pass

    @property
    def random_gen(self):
        pass

    def new_seed(self):
        pass

    def load_checkpoint(self):
        pass

    def save_checkpoint(self):
        pass

    def seed_known_random_generators(self):
        pass

    def move_to_correct_device(self):
        pass

    Counter = object()

    Recorder = object()

    Looper = object()

    BetterDataLoader = object()

    experiment = object()

    parameter = object()


class TorchContext:

    def __init__(self):
        self._items = {}

    def __getattr__(self, name):
        if name in self._items:
            return self._items[name].handle
        
        raise AttributeError()

    def __setattr__(self, name, value):
        self.add(name, value)

    def add(self, name, item, **kwargs):
        if not name.isidentifier():
            raise ValueError(f"'{name}' is not a valid python identifier.")

        if hasattr(self, name):
            raise ValueError(f"'{name}' is not a valid identifier because it is used by the context object itself.")
        


    def state_dict(self):
        pass

    def load_state_dict(self, state):
        pass

    def cuda(self):
        pass

    def distributed(self):
        pass

@dataclasses.dataclass
class ContextItem:
    """Represents an item registered with a pytorch context. It is just a
    wrapper around an object together with some configuration items."""
    
    handle: typing.Any # A handle to the actual item / object

    cuda: bool = True # Can be used to disable cuda movement for specific items
    checkpoint: bool = True # Set to False to exclude this item from being saved and loaded from checkpoints




global_state = TorchExperiment()
