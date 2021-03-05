import collections
import dataclasses
import logging
import pathlib
import random
import typing

import numpy as np
import torch

logger = logging.getLogger('PytorchAPI')

class ReadOnlyDict(collections.Mapping):

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key): 
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class PystematicPytorchAPI:

    def __init__(self):
        self._params = None
        self._output_dir = None
        self._random_gen = None
        self._ctx = None
    
    def _initialize(self):
        """This is an internal function used to initialize the api object when a
        new experiment is started"""
        pass

    @property
    def output_dir(self) -> pathlib.Path:
        """Returns a pathlib.Path object that points to the current output
        directory"""
        return self._output_dir

    @property
    def random_gen(self):
        """Holds the global random generator used to generate new seeds"""
        return self._random_gen

    @property
    def params(self) -> dict:
        """Holds the parameters of the current experiment"""
        return ReadOnlyDict(self._params)
    
    @property
    def ctx(self):
        """Holds the context object for the current experiment. The type of this
        object depends on the type of the current experiment"""
        return self._ctx

    def new_seed(self, nbits=32) -> int:
        """Use this function to generate random numbers seeded by the experiment
        parameter `random_seed`. Expected use is to seed your own random number
        generators.
        """
        return self.random_gen.getrandbits(nbits)

    def seed_known_random_generators(self):
        """This is just a helper to seed all known random modules with reproducible seeds."""
        
        logger.info(f"Seeding random modules for python, numpy and pytorch.")

        random.seed(self.new_seed())
        torch.manual_seed(self.new_seed())
        np.random.seed(self.new_seed())

    def launch_subprocess(self, additional_params):
        """Launches a subprocess. The subprocess will have the same output
        directory and parameters as the current process"""
        pass

    def is_subprocess(self) -> bool:
        """Returns true if this process is a subprocess"""
        pass

    #
    # Helpers
    #
    def place_on_correct_device(self, *args):
        """Utility method to place a batch of data on the correct device (i.e.
        cuda or cpu) depending on the 'cuda' experiment parameter."""
        res = []
        for arg in args:
            if self.params["cuda"] and callable(getattr(arg, "cuda", None)):
                res.append(arg.cuda())
            else:
                res.append(arg)
        return res

    #
    # Pytorch distributed
    #

    def init_distributed(self):
        pass
    
    def is_distributed(self):
        return torch.distributed.is_initialized()

    def is_master(self):
        return not self.is_distributed() or self.get_rank() == 0

    def get_num_processes(self):
        if self.is_distributed():
            return torch.distributed.get_world_size()
            
        return 1

    def get_allocated_device_ids(self):
        if self.is_distributed():
            return [self.params["local_rank"]]
        
        return [i for i in range(torch.cuda.device_count())]

    def get_rank(self):
        if self.is_distributed():
            return torch.distributed.get_rank()
        
        return 0
    
    def broadcast_from_master(self, value):
        value = torch.tensor(value)

        if self.is_distributed():
            torch.distributed.broadcast(value, 0)

        return value

    def distributed_barrier(self):
        if self.is_distributed():
            torch.distributed.barrier()

    #
    # Checkpoints
    #

    def save_checkpoint(self, filename):
        pass

    def load_checkpoint(self, file):
        pass

    def restore_from_checkpoint(self):
        pass





@dataclasses.dataclass
class ContextItem:
    """Represents an item registered with a pytorch context. It is just a
    wrapper around an object together with some configuration items."""
    
    handle: typing.Any # A handle to the actual item / object

    cuda: bool = True # Can be used to disable cuda movement for specific items
    checkpoint: bool = True # Set to False to exclude this item from being saved and loaded from checkpoints


class TorchContext:

    def __init__(self):
        self._items = {}
        self._cuda = False
        self._distributed = False
        self._checkpoint = {}

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


