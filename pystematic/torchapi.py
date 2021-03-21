import dataclasses
import datetime
import logging
import multiprocessing
import pathlib
import random
import typing
import string
import time
import atexit

import numpy as np
import torch
import wrapt
import yaml
import tqdm

from .recording import Recorder
from .click_adapter import invoke_experiment_with_parsed_args, get_current_experiment
from . import utils


logger = logging.getLogger('pystematic_torch')


#
# General
#

params: dict = wrapt.ObjectProxy(None)


output_dir: pathlib.Path = wrapt.ObjectProxy(None)


params_file: pathlib.Path = wrapt.ObjectProxy(None)
"""Points to the current parameters file."""


random_gen: random.Random = wrapt.ObjectProxy(None)
"""Holds the global random generator used to generate new seeds"""


context = wrapt.ObjectProxy(None)


def new_seed(nbits=32) -> int:
    """Use this function to generate random numbers seeded by the experiment
    parameter ``random_seed``. Expected use is to seed your own random number
    generators.
    """
    return random_gen.getrandbits(nbits)


def seed_known_random_generators() -> None:
    """This is just a helper to seed all known random modules with
    reproducible seeds."""

    logger.info(f"Seeding random modules for python, numpy and pytorch.")

    random.seed(new_seed())
    torch.manual_seed(new_seed())
    np.random.seed(new_seed())


def run_parameter_sweep(experiment, list_of_params, max_num_processes=1, num_gpus_per_process=None) -> None:
    """Runs an experiment with a set of different params. At most
    :obj:`max_num_processes` concurrent processes will be used.
    """

    pool = utils.ProcessQueue(max_num_processes, range(torch.cuda.device_count()))
    pool.run_and_wait_for_completion(experiment, list_of_params)


def run_experiment(experiment, **params) -> multiprocessing.Process:
    """Runs an experiment in a new process.

    Args:
        experiment: A handle to the experiment to run. 
        **params: The parameters to pass to the experiment.

    Returns:
        A handle to the process object. Not that the process is started before
        it is returned.
    """

    logger.debug(f"Running experiment '{experiment.experiment_name}' with arguments {params}.")

    proc = multiprocessing.get_context('spawn').Process(
        target=invoke_experiment_with_parsed_args,
        args=(experiment, params)
    )

    proc.start()

    return proc


def launch_subprocess(**additional_params) -> multiprocessing.Process:
    """Launches a subprocess. The subprocess will have the same output
    directory and parameters as the current process.

    Args:
        **additional_params: Any additional parameters that should be 
            passed to the subprocess. Params given here takes precedence 
            over the parameters copied from the current experiment.

    .. warning:: 

        The subprocess will be initialized with the same random
        seed as the current process. If this is not what you want, you
        should pass a new seed to this function in the ``random_seed`` parameter. 

        E.g.:
        >>> pystematic.launch_subprocess(random_seed=pystematic.new_seed())

    """
    logger.debug("Launching subprocess...")

    subprocess_params = {name: value for name, value in params.items()}

    for name, value in additional_params.items():
        subprocess_params[name] = value

    subprocess_params["subprocess"] = str(params_file)

    logger.debug(f"Launching subprocess with arguments '{' '.join(subprocess_params)}'.")

    proc = multiprocessing.Process(
        target=invoke_experiment_with_parsed_args,
        args=(get_current_experiment(), subprocess_params)
    )
    proc.start()

    return proc


def is_subprocess() -> bool:
    """Returns true if this process is a subprocess. I.e. it has been
    launched by a call to :func:`launch_subprocess` in a parent process.
    """

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


def iterate(iterable):
    """Returns a wrapper around the iterator that show a progessbar (tqdm).
    The progessbar is silenced in non-master processes.
    """

    if is_master():
        return tqdm.tqdm(iterable, leave=True)

    return iterable


#
# Pytorch distributed
#

def init_distributed() -> None:
    """Initializes a distributed runtime. This function is called automatically 
    during initialization if the parameter ``distributed`` is set to ``True``.
    """
    if params["local_rank"] is None:
        for i in range(1, params["nproc_per_node"]):
            launch_subprocess(local_rank=i)

        local_rank = 0
    else:
        local_rank = params["local_rank"]

    global_rank = params["nproc_per_node"] * params["node_rank"] + local_rank
    world_size = params["nproc_per_node"] * params["nnodes"]

    logger.debug(f"Initializing distributed runtime (world size '{world_size}', "
                  "local rank '{local_rank}', global rank '{global_rank}')...")

    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f"tcp://{params['master_addr']}:{params['master_port']}",
        world_size=world_size,
        rank=global_rank
    )

    logger.debug(f"Distributed runtime initialized.")


def is_distributed() -> bool:
    return torch.distributed.is_initialized()


def is_master() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def get_num_processes() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    return 1


def get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    return 0


def broadcast_from_master(value):
    value = torch.tensor(value)

    if torch.distributed.is_initialized():
        torch.distributed.broadcast(value, 0)

    return value


def distributed_barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


#
# Checkpoints
#

def save_checkpoint(filename) -> None:
    """Saves registered items to a file. All items that have a function named
    ``state_dict`` will be saved by calling that function and saving the
    returned value. This function will make sure to only save the checkpoint in
    the master process when called in distributed mode.
    """

    if is_master():
        checkpoint_file_path = output_dir.joinpath(filename)

        logger.info(f"Saving checkpoint '{checkpoint_file_path}'.")

        with checkpoint_file_path.open("wb") as f:
            torch.save(context.state_dict(), f)


def load_checkpoint(checkpoint_file_path) -> dict:
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

    handle: typing.Any  # A handle to the actual item / object

    cuda: bool = True  # Can be used to disable cuda movement for specific items
    
    checkpoint: bool = True # Set to False to exclude this item from being saved and loaded from checkpoints


class TorchContext:
    """A context object that helps with managing the different kinds of objects
    you need during a pytorch session.

    A context object is used to gather all the different types of object you
    need during a pytorch training session. It also help you with transparently
    transform the objects you add to accomodate for things such as use of cuda,
    use of distributed models.

    A context object makes use of the following parameters to determine how
    objects should be handled:

    - cuda
    - distributed

    """

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
        """Adds item :param:`item` to the context with name :param:`name`.

        Args:
            name (str): The name that the item will be accessible under
            item (any): The item to add
            cuda (bool, optional): Whether the item should be automatically 
                moved to cuda when added. Defaults to True.
            checkpoint (bool, optional): Controls whether the item should be 
                saved and read to/from checkpoints. Defaults to True.

        """
        if not name.isidentifier():
            raise ValueError(f"'{name}' is not a valid python identifier.")

        if name in dir(self):
            raise ValueError(f"'{name}' is not allowed as an identifier because it "
                              "is used by the context object itself.")

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
# Internal
#

def _initialize(_params):
    """This is an internal function used to initialize the api object when a
    new experiment is started"""

    if _params["debug"]:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    logging.basicConfig(level=log_level, handlers=[utils.PytorchLogHandler()])

    params.__wrapped__ = _params

    if params["subprocess"]:
        logger.debug(f"Initializing subprocess...")
        output_dir.__wrapped__ = pathlib.Path(params["subprocess"]).parent
        params_file.__wrapped__ = pathlib.Path(params["subprocess"])
    else:
        output_dir.__wrapped__ = _create_log_dir_name(params["output_dir"], get_current_experiment().experiment_name)
        output_dir.__wrapped__.mkdir(parents=True, exist_ok=True)
        params_file.__wrapped__ = output_dir.joinpath("parameters.yml")

        logger.debug(f"Writing parameters file to '{params_file}'.")
        with params_file.open("w") as f:
            yaml.dump(_params, f, default_flow_style=False)

    random_gen.__wrapped__ = random.Random(params["random_seed"])

    context.__wrapped__ = TorchContext()

    if params["checkpoint"]:
        logger.info(f"Loading checkpoint '{params['checkpoint']}'.")
        context.load_state_dict(load_checkpoint(params["checkpoint"]))

    if params["distributed"]:
        init_distributed()


def _cleanup():
    procs = multiprocessing.active_children()
    for proc in procs:
        try:
            proc.terminate()
        except Exception:
            pass

    for proc in procs:
        try:
            proc.join()
        except Exception:
            pass


atexit.register(_cleanup)

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
    directory = pathlib.Path(output_dir).resolve().joinpath(experiment_name).joinpath(current_time)

    if directory.exists():
        chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
        suffix = "".join(random.SystemRandom().choice(chars) for _ in range(6))
        directory = directory.with_name(f"{directory.name}-{suffix}")

    return directory

