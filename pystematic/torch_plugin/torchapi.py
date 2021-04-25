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
from .core import get_current_experiment
from . import utils
from . import yaml_wrapper as yaml


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

    pool = utils.ProcessQueue(max_num_processes, range(torch.cuda.device_count()), num_gpus_per_process)
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

    return experiment.run_in_new_process(params)


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
    subprocess_params = {name: value for name, value in params.items()}

    for name, value in additional_params.items():
        subprocess_params[name] = value

    subprocess_params["subprocess"] = str(params_file)

    logger.debug(f"Launching subprocess with arguments '{' '.join(subprocess_params)}'.")

    return get_current_experiment().run_in_new_process(subprocess_params)


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
                 f"local rank '{local_rank}', global rank '{global_rank}')...")

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

def save_checkpoint(ctx, filename) -> None:
    """Saves registered items to a file. All items that have a function named
    ``state_dict`` will be saved by calling that function and saving the
    returned value. This function will make sure to only save the checkpoint in
    the master process when called in distributed mode.
    """

    if is_master():
        checkpoint_file_path = output_dir.joinpath(filename)

        logger.info(f"Saving checkpoint '{checkpoint_file_path}'.")

        with checkpoint_file_path.open("wb") as f:
            torch.save(ctx.state_dict(), f)


def load_checkpoint(checkpoint_file_path) -> dict:
    """Loads and returns a checkpoint from the given filepath."""
    with open(checkpoint_file_path, "rb") as f:
        return torch.load(f, map_location="cpu")


#
# Context
#

    
#
# Internal
#

def _initialize(_params):
    """This is an internal function used to initialize the api object when a
    new experiment is started"""

    params.__wrapped__ = _params
    
    if params["debug"]:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    logging.basicConfig(level=log_level, handlers=[utils.PytorchLogHandler()])

    if params["subprocess"]:
        logger.debug(f"Initializing subprocess...")
        output_dir.__wrapped__ = pathlib.Path(params["subprocess"]).parent
        params_file.__wrapped__ = pathlib.Path(params["subprocess"])
    else:
        output_dir.__wrapped__ = _create_log_dir_name(params["output_dir"], get_current_experiment().name)
        output_dir.__wrapped__.mkdir(parents=True, exist_ok=True)
        params_file.__wrapped__ = output_dir.joinpath("parameters.yml")

        logger.debug(f"Writing parameters file to '{params_file}'.")
        with params_file.open("w") as f:
            yaml.dump(_params, f)

    random_gen.__wrapped__ = random.Random(params["random_seed"])

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
def _create_log_dir_name(output_dir, experiment_name):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    directory = pathlib.Path(output_dir).resolve().joinpath(experiment_name).joinpath(current_time)

    if directory.exists():
        chars = string.digits + string.ascii_lowercase # + string.ascii_uppercase
        suffix = "".join(random.SystemRandom().choice(chars) for _ in range(6))
        directory = directory.with_name(f"{directory.name}-{suffix}")

    return directory

