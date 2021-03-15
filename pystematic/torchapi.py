import dataclasses
import datetime
import logging
import multiprocessing
import pathlib
import random
import typing

import click
import numpy as np
import torch
import wrapt
import yaml

from .recording import Recorder

logger = logging.getLogger('pystematic_torch')

#
# Internal
#

def _initialize(self, params):
    """This is an internal function used to initialize the api object when a
    new experiment is started"""

    if params["debug"]:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    logging.basicConfig(level=log_level, handlers=[PytorchLogHandler()])


    self._params.__wrapped__ = params

    if params["subprocess"]:
        logger.debug(f"Initializing subprocess...")
        self._output_dir.__wrapped__ = pathlib.Path(params["subprocess"]).parent
        self._params_file.__wrapped__ = pathlib.Path(params["subprocess"])
    else:
        self._output_dir.__wrapped__ = _create_log_dir_name(params["output_dir"], click.get_current_context().command.name)
        self._output_dir.__wrapped__.mkdir(parents=True, exist_ok=True)
        self._params_file.__wrapped__ = self._output_dir.joinpath("parameters.yml")

        logger.debug(f"Writing parameters file to '{self.params_file}'.")
        with self.params_file.open("w") as f:
            yaml.dump(params, f, default_flow_style=False)

    self._random_gen.__wrapped__ = random.Random(params["random_seed"])

    self._context.__wrapped__ = TorchContext()
    

    if params["checkpoint"]:
        logger.info(f"Loading checkpoint '{params['checkpoint']}'.")
        self._context.load_state_dict(self.load_checkpoint(params["checkpoint"]))

    if params["distributed"]:
        self.init_distributed()


#
# General
#

params: dict = wrapt.ObjectProxy(None)
"""Holds the parameters of the current experiment"""


output_dir: pathlib.Path = wrapt.ObjectProxy(None)
"""Returns a pathlib.Path object that points to the current output
directory. All output from an experiment should be written to this
folder.
"""

params_file: pathlib.Path  = wrapt.ObjectProxy(None)
"""The current parameters file."""


random_gen: random.Random = wrapt.ObjectProxy(None)
"""Holds the global random generator used to generate new seeds"""


context = wrapt.ObjectProxy(None)
"""Holds the context object for the current experiment."""


def new_seed(nbits=32) -> int:
    """Use this function to generate random numbers seeded by the experiment
    parameter `random_seed`. Expected use is to seed your own random number
    generators.
    """
    return random_gen.getrandbits(nbits)


def seed_known_random_generators():
    """This is just a helper to seed all known random modules with
    reproducible seeds."""
    
    logger.info(f"Seeding random modules for python, numpy and pytorch.")

    random.seed(new_seed())
    torch.manual_seed(new_seed())
    np.random.seed(new_seed())


def run_experiment(experiment, **params):
    """Runs an experiment in a new process
    """

    logger.debug(f"Running experiment '{experiment.name}' with arguments {params}.")

    proc = multiprocessing.Process(
        target=_invoke_command_with_parsed_args, 
        args=(experiment, params)
    )

    proc.start()

    return proc


def launch_subprocess(**additional_params):
    """Launches a subprocess. The subprocess will have the same output
    directory and parameters as the current process

    .. warning:: 

        The subprocess will be initialized with the same random
        seed as the current process. If this is not what you want, you
        should pass a new seed to this function in the ``random_seed`` parameter. 

        E.g: :code:`ps.launch_subprocess(random_seed=ps.new_seed())`

    """
    logger.debug("Launching subprocess...")

    params = {name:value for name, value in params.items()}

    for name, value in additional_params.items():
        params[name] = value

    params["subprocess"] = str(params_file)

    logger.debug(f"Launching subprocess with arguments '{' '.join(params)}'.")
        
    experiment = click.get_current_context().command

    proc = multiprocessing.Process(
        target=_invoke_command_with_parsed_args, 
        args=(experiment, params)
    )

    proc.start()

    return proc


def is_subprocess() -> bool:
    """Returns true if this process is a subprocess. I.e. it has been
    launched by a call to :func:`launch_subprocess` in a parent process."""
    return params["subprocess"] is not None


#
# Helpers
#

def place_on_correct_device(*args):
    """Utility method to place a batch of data on the correct device (i.e.
    cuda or cpu) depending on the 'cuda' experiment parameter."""
    res = []
    for arg in args:
        if params["cuda"] and callable(getattr(arg, "cuda", None)):
            res.append(arg.cuda())
        else:
            res.append(arg)
    return res


#
# Pytorch distributed
#

def init_distributed():
    if params["local_rank"] is None:
        for i in range(1, params["nproc_per_node"]):
            launch_subprocess(local_rank=i)
        
        local_rank = 0
    else:
        local_rank = params["local_rank"]

    global_rank = params["nproc_per_node"] * params["node_rank"] + local_rank
    world_size = params["nproc_per_node"] * params["nnodes"]

    logger.debug(f"Initializing distributed runtime (world size '{world_size}', local rank '{local_rank}', global rank '{global_rank}')...")
    
    torch.cuda.set_device(local_rank)
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f"tcp://{params['master_addr']}:{params['master_port']}",
        world_size=world_size,
        rank=global_rank
    )

    logger.debug(f"Distributed runtime initialized.")


def is_distributed():
    return torch.distributed.is_initialized()


def is_master():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def get_num_processes():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
        
    return 1


def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    
    return 0


def broadcast_from_master(value):
    value = torch.tensor(value)

    if torch.distributed.is_initialized():
        torch.distributed.broadcast(value, 0)

    return value


def distributed_barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


#
# Checkpoints
#

def save_checkpoint(filename):
    """Saves registered items to a file. All items that have a function
    named ``state_dict`` will be saved by calling that function and saving the
    returned value.
    """

    if is_master():
        checkpoint_file_path = output_dir.joinpath(filename)

        logger.info(f"Saving checkpoint '{checkpoint_file_path}'.")

        with checkpoint_file_path.open("wb") as f:
            torch.save(context.state_dict(), f)


def load_checkpoint(checkpoint_file_path):
    """Loads and returns a checkpoint from the given filepath."""
    with open(checkpoint_file_path, "rb") as f:
        return torch.load(f, map_location="cpu")


#
# Context
#

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
        
        raise AttributeError(f"TorchContext does not have an attribute named '{name}'.")

    def __setattr__(self, name, value):
        self.add(name, value)

    def has(self, name):
        return name in self._items

    def add(self, name, item, cuda=True, checkpoint=True):
        if not name.isidentifier():
            raise ValueError(f"'{name}' is not a valid python identifier.")

        if name in dir(self):
            raise ValueError(f"'{name}' is not allowed as an identifier because it is used by the context object itself.")
        
        if name in self._checkpoint:
            if checkpoint:
                
                logger.debug(f"Loading parameters from checkpoint for '{name}'.")
                item = _load_state_dict_into_item(item, self._checkpoint[name])
            else:
                logger.debug(f"Not loading item '{name}' from checkpoint due to item specific config.")
        
        if params["cuda"] and callable(getattr(item, "cuda", None)):
            if cuda:
                logger.debug(f"Moving '{name}' to CUDA.")
                item = item.cuda()
            else:
                logger.debug(f"Not moving item '{name}' to cuda due to item specific config.")
        
        
        if isinstance(item, torch.nn.Module):
            if params["distributed"] and any([p.requires_grad for p in item.parameters()]):
                logger.debug(f"Converting to distributed for model '{name}'.")
                
                item = torch.nn.parallel.DistributedDataParallel(
                    module=item,
                    device_ids=[torch.cuda.current_device()]
                )

        elif isinstance(item, Recorder):
            item.set_output_dir(output_dir)

            if not is_master():
                item.silence()
        
        self._items[name] = ContextItem(handle=item, cuda=cuda, checkpoint=checkpoint)

    def state_dict(self):
        dict_with_state = {
            "__torch_version": torch.__version__
        }

        _supported_types = (int, float, complex, str)

        for name, item in self._items.items():
            
            if item.checkpoint:
                if callable(getattr(item.handle, "state_dict", None)):
                    if isinstance(item.handle, torch.nn.parallel.DistributedDataParallel):
                        dict_with_state[name] = item.handle.module.state_dict()
                    else:
                        dict_with_state[name] = item.handle.state_dict()
                elif isinstance(item.handle, _supported_types):
                    dict_with_state[name] = {
                        "native_value": item.handle
                    }
            else:
                logger.debug(f"Not saving item '{name}' to checkpoint due to item specific config.")
                    

        return dict_with_state

    def load_state_dict(self, state):
        object.__setattr__(self, "_checkpoint", state)

        for name, item in self._items.items():
            if item.checkpoint:
                item.handle = _load_state_dict_into_item(item.handle, self._checkpoint[name])
            else:
                logger.debug(f"Not loading item '{name}' from checkpoint due to item specific config.")


#
# private Helpers
#

_supported_types = (int, float, complex, str)


def _load_state_dict_into_item(item, state_dict):

    if callable(getattr(item, "load_state_dict", None)):
        if isinstance(item, torch.nn.parallel.DistributedDataParallel):
            item.module.load_state_dict(_move_to_same_device_as(state_dict, item.module))
        else:
            item.load_state_dict(_move_to_same_device_as(state_dict, item))

    elif isinstance(item, _supported_types):
        return state_dict["native_value"]
    else:
        logger.debug(f"Cannot checkpoint object of type '{type(item)}'")

    return item


def _get_item_state_dict(item):
    if callable(getattr(item, "state_dict", None)):
        if isinstance(item, torch.nn.parallel.DistributedDataParallel):
            return item.module.state_dict()
        else:
            return item.state_dict()
    elif isinstance(item, _supported_types):
        return {
            "native_value": item
        }
    else:
        logger.debug(f"Cannot checkpoint object of type '{type(item)}'")
    
    return None


def _move_to_same_device_as(to_move, target):
    if hasattr(target, "device"):
        return _move_to_device(to_move, target.device)
    
    elif callable(getattr(target, "parameters", None)):
        try:
            return _move_to_device(to_move, next(target.parameters()).device)
        except StopIteration:
            pass

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


def _create_log_dir_name(output_dir, experiment_name):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return pathlib.Path(output_dir).resolve().joinpath(experiment_name).joinpath(current_time)


def _invoke_command_with_parsed_args(command, args_dict):
    extra = {}
    for key, value in command.context_settings.items():
        if key not in extra:
            extra[key] = value

    ctx = click.Context(command, **extra)

    for param in command.get_params(ctx):
        value, _ = param.handle_parse_result(ctx, args_dict, None)

    with ctx as ctx:
        command.invoke(ctx)


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
