import collections
import dataclasses
import datetime
import logging
import os
import pathlib
import random
import subprocess
import sys
import typing

import numpy as np
import torch
import yaml
import click

from .recording import Recorder

logger = logging.getLogger('PytorchAPI')


class PytorchLogHandler(logging.Handler):
    """Handle logging for both single- and multiprocess contexts."""

    def __init__(self):
        super().__init__()
        self._colors = {
            'DEBUG':    'magenta',
            'INFO':     'blue',
            'WARNING':  'yellow',
            'ERROR':    'red'
        }

    def handle(self, record):
        level = click.style(f"[{record.levelname}]", fg=self._colors[record.levelname])  
        msg = click.style(f"{record.getMessage()}", fg="white")

        name = click.style(f"[{record.name}]", fg="green")
        
        if torch.distributed.is_initialized():
            rank = click.style(f"[RANK {torch.distributed.get_rank()}]", fg="green")
            click.echo(f"{level} {rank} {name} {msg}")
        else:
            click.echo(f"{level} {name} {msg}")

class PystematicPytorchAPI:

    def __init__(self):
        self._params = None
        self._output_dir = None
        self._random_gen = None
        self._ctx = None
    
    def _initialize(self, params):
        """This is an internal function used to initialize the api object when a
        new experiment is started"""

        if params["debug"]:
            log_level = "DEBUG"
        else:
            log_level = "INFO"

        logging.basicConfig(level=log_level, handlers=[PytorchLogHandler()])


        self._params = params

        if params["subprocess"]:
            self._output_dir = pathlib.Path(params["subprocess"]).parent
        else:
            self._output_dir = _create_log_dir_name(params["output_dir"], params["experiment_name"])
            self._output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Writing parameters file to '{self.params_file}'.")
            with self.params_file.open("w") as f:
                yaml.dump(params, f, default_flow_style=False)

        self._random_gen = random.Random(params["random_seed"])


        self._ctx = TorchContext()

        if params["checkpoint"]:
            self._ctx.load_state_dict(self.load_checkpoint(params["checkpoint"]))

        if params["distributed"]:
            self.init_distributed()

    @property
    def output_dir(self) -> pathlib.Path:
        """Returns a pathlib.Path object that points to the current output
        directory"""
        return self._output_dir

    @property
    def params_file(self) -> pathlib.Path:
        return self._output_dir.joinpath("parameters.yml")

    @property
    def random_gen(self):
        """Holds the global random generator used to generate new seeds"""
        return self._random_gen

    @property
    def params(self) -> dict:
        """Holds the parameters of the current experiment"""
        return self._params
    
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
        """This is just a helper to seed all known random modules with
        reproducible seeds."""
        
        logger.info(f"Seeding random modules for python, numpy and pytorch.")

        random.seed(self.new_seed())
        torch.manual_seed(self.new_seed())
        np.random.seed(self.new_seed())

    def launch_subprocess(self, **additional_params):
        """Launches a subprocess. The subprocess will have the same output
        directory and parameters as the current process

        .. warning:: 

            The subprocess will be initialized with the same random
            seed as the current process. If this is not what you want, you
            should pass a new seed to this function in the :param:`random_seed` parameter. 

            E.g:``ps.launch_subprocess(random_seed=ps.new_seed())``

        """
        logger.debug("Launching subprocess...")

        cmd = [sys.executable] + sys.argv[:]
        cmd.append("--subprocess")
        cmd.append(str(self.params_file))

        for name, value in additional_params.items():
            cmd.append(_param_name_to_cli_name(name))
            cmd.append(str(value))
            
        logger.debug(f"Executing '{' '.join(cmd)}'.")

        process = subprocess.Popen(cmd, env=os.environ.copy())

    def is_subprocess(self) -> bool:
        """Returns true if this process is a subprocess. I.e. it has been
        launched by a call to :func:`launch_subprocess` in a parent process."""
        return self.params["subprocess"] is not None

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
        if self.params["local_rank"] is None:
            for i in range(1, self.params["nproc_per_node"]):
                self.launch_subprocess(local_rank=i)
            
            local_rank = 0

        global_rank = self.params["nproc_per_node"] * self.params["node_rank"] + local_rank
        world_size = self.params["nproc_per_node"] * self.params["nnodes"]

        logger.debug(f"Initializing distributed runtime (world size '{world_size}', local rank '{local_rank}', global rank '{global_rank}')...")
        
        torch.cuda.set_device(local_rank)
        
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=f"tcp://{self.params['master_addr']}:{self.params['master_port']}",
            world_size=world_size,
            rank=global_rank
        )

        logger.debug(f"Distributed runtime initialized.")
    
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
            if self.params["local_rank"]:
                return [self.params["local_rank"]]
            
            return [0]
        
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
        """Saves registered items to a file. All items that have a function
        named 'state_dict' will be saved by calling that function and saving the
        returned value.
        """

        if self.is_master():
            checkpoint_file_path = self.output_dir.joinpath(filename)

            logger.info(f"Saving checkpoint '{checkpoint_file_path}'.")

            with checkpoint_file_path.open("wb") as f:
                torch.save(self.ctx.state_dict(), f)

    def load_checkpoint(self, checkpoint_file_path):
        """Loads and returns a checkpoint from the given filepath."""
        with open(checkpoint_file_path, "rb") as f:
            return torch.load(f, map_location="cpu")


api = PystematicPytorchAPI()


@dataclasses.dataclass
class ContextItem:
    """Represents an item registered with a pytorch context. It is just a
    wrapper around an object together with some configuration items."""
    
    handle: typing.Any # A handle to the actual item / object

    cuda: bool = True # Can be used to disable cuda movement for specific items
    checkpoint: bool = True # Set to False to exclude this item from being saved and loaded from checkpoints


class TorchContext:

    def __init__(self):
        object.__setattr__(self, "_items", {})
        object.__setattr__(self, "_checkpoint", {})

    def __getattr__(self, name):
        if name in self._items:
            return self._items[name].handle
        
        raise AttributeError()

    def __setattr__(self, name, value):
        self.add(name, value)

    def add(self, name, item, cuda=True, checkpoint=True):
        if not name.isidentifier():
            raise ValueError(f"'{name}' is not a valid python identifier.")

        if hasattr(self, name):
            raise ValueError(f"'{name}' is not a valid identifier because it is used by the context object itself.")
        
        if name in self._checkpoint:
            if checkpoint:
                logger.debug(f"Loading parameters from checkpoint for '{name}'.")
                item.load_state_dict(_move_to_same_device_as(self._checkpoint[name], item))
            else:
                logger.debug(f"Not loading item '{name}' from checkpoint due to item specific config.")
        
        if api.params["cuda"] and callable(getattr(item, "cuda", None)):
            if cuda:
                logger.debug(f"Moving '{name}' to CUDA.")
                item = item.cuda()
            else:
                logger.debug(f"Not moving item '{name}' to cuda due to item specific config.")
        
        
        if isinstance(item, torch.nn.Module) and any([p.requires_grad for p in item.parameters()]):
            if api.params["distributed"]:
                logger.debug(f"Converting to distributed for model '{name}'.")
                
                item = torch.nn.parallel.DistributedDataParallel(
                    module=item,
                    device_ids=api.get_allocated_device_ids(),
                    output_device=api.get_allocated_device_ids()[0],
                    broadcast_buffers=True
                )

        elif isinstance(item, Recorder):
            item.set_output_dir(api.output_dir)

            if not api.is_master():
                item.silence()
        
        self._items[name] = ContextItem(handle=item, cuda=cuda, checkpoint=checkpoint)

    def state_dict(self):
        dict_with_state = {
            "__torch_version": torch.__version__
        }

        for name, item in self._items.items():
            if callable(getattr(item.handle, "state_dict", None)):
                if item.checkpoint:
                    if isinstance(item.handle, torch.nn.parallel.DistributedDataParallel):
                        dict_with_state[name] = item.handle.module.state_dict()
                    else:
                        dict_with_state[name] = item.handle.state_dict()
                else:
                    logger.debug(f"Not saving item '{name}' to checkpoint due to item specific config.")

        return dict_with_state

    def load_state_dict(self, state):
        object.__setattr__(self, "_checkpoint", state)

        for name, item in self._items.items():
            if callable(getattr(item.handle, "load_state_dict", None)):
                if item.checkpoint:
                    if isinstance(item.handle, torch.nn.parallel.DistributedDataParallel):
                        item.handle.module.load_state_dict(_move_to_same_device_as(self._checkpoint[name], item.handle.module))
                    else:
                        item.handle.load_state_dict(_move_to_same_device_as(self._checkpoint[name], item.handle))
                else:
                    logger.debug(f"Not loading item '{name}' from checkpoint due to item specific config.")

#
# Helpers
#

def _move_to_same_device_as(to_move, target):
    if hasattr(target, "device"):
        return _move_to_device(to_move, target.device)
    
    elif callable(getattr(target, "parameters", None)):
        return _move_to_device(to_move, next(target.parameters()).device)

    return to_move


def _move_to_device(obj, device):
    if isinstance(obj, dict):
        res = {}
        for name, value in obj.items():
            res[name] = _move_to_device(value, device)

    elif isinstance(obj, list) or isinstance(obj, tuple):
        res = []
        for i in range(len(obj)):
            res.append(_move_to_device(obj[i]))

    elif callable(getattr(obj, "to", None)):
        res = obj.to(device=device)

    else:
        raise Exception(f"Unsupported object type '{type(obj)}'")

    return res


def _param_name_to_cli_name(name):
    return f"--{name.replace('_', '-')}"


def _create_log_dir_name(output_dir, experiment_name):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return pathlib.Path(output_dir).resolve().joinpath(experiment_name).joinpath(current_time)
