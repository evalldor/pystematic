import logging
import functools
import pathlib
import torch
import tqdm

from . import utils
from ..classic_plugin import ClassicApi, classic_params

from pystematic.pluginapi import (
    register_plugin,
    experiment_decorator, 
    group_decorator, 
    parameter_decorator
)

from . import context, recording, torchutil

import pystematic.core as core

logger = logging.getLogger('pystematic_torch')

class TorchApi(ClassicApi):
    
    def __init__(self) -> None:
        super().__init__()

    def init_experiment(self, experiment, params):
        if params["debug"]:
            log_level = "DEBUG"
        else:
            log_level = "INFO"

        logging.basicConfig(level=log_level, handlers=[utils.PytorchLogHandler()])

        super().init_experiment(experiment, params)

        if self.params["distributed"]:
            self.init_distributed()

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

    def iterate(self, iterable):
        """Returns a wrapper around the iterator that show a progessbar (tqdm).
        The progessbar is silenced in non-master processes.
        """

        if self.is_master():
            return tqdm.tqdm(iterable, leave=True)

        return iterable

    def save_checkpoint(self, ctx, filename) -> None:
        """Saves registered items to a file. All items that have a function named
        ``state_dict`` will be saved by calling that function and saving the
        returned value. This function will make sure to only save the checkpoint in
        the master process when called in distributed mode.
        """

        if self.is_master():
            checkpoint_file_path = self.output_dir.joinpath(filename)

            logger.info(f"Saving checkpoint '{checkpoint_file_path}'.")

            with checkpoint_file_path.open("wb") as f:
                torch.save(ctx.state_dict(), f)

    def load_checkpoint(self, checkpoint_file_path) -> dict:
        """Loads and returns a checkpoint from the given filepath."""
        with open(checkpoint_file_path, "rb") as f:
            return torch.load(f, map_location="cpu")

    def run_parameter_sweep(self, experiment, list_of_params, max_num_processes=1, num_gpus_per_process=None) -> None:
        """Runs an experiment with a set of different params. At most
        :obj:`max_num_processes` concurrent processes will be used.
        """

        pool = utils.ProcessQueue(max_num_processes, range(torch.cuda.device_count()), num_gpus_per_process)
        pool.run_and_wait_for_completion(experiment, list_of_params)

    #
    # Pytorch distributed data parallell
    #

    def init_distributed(self) -> None:
        """Initializes a distributed runtime. This function is called automatically 
        during initialization if the parameter ``distributed`` is set to ``True``.
        """
        if self.params["local_rank"] is None:
            for i in range(1, self.params["nproc_per_node"]):
                self.launch_subprocess(local_rank=i)

            local_rank = 0
        else:
            local_rank = self.params["local_rank"]

        global_rank = self.params["nproc_per_node"] * self.params["node_rank"] + local_rank
        world_size = self.params["nproc_per_node"] * self.params["nnodes"]

        logger.debug(f"Initializing distributed runtime (world size '{world_size}', "
                    f"local rank '{local_rank}', global rank '{global_rank}')...")

        torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(
            backend='nccl',
            init_method=f"tcp://{self.params['master_addr']}:{self.params['master_port']}",
            world_size=world_size,
            rank=global_rank
        )

        logger.debug(f"Distributed runtime initialized.")

    def is_distributed(self) -> bool:
        return torch.distributed.is_initialized()

    def is_master(self) -> bool:
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    def get_num_processes(self) -> int:
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()

        return 1

    def get_rank(self) -> int:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()

        return 0

    def broadcast_from_master(self, value):
        value = torch.tensor(value)

        if torch.distributed.is_initialized():
            torch.distributed.broadcast(value, 0)

        return value

    def distributed_barrier(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


pytorch_params = [
    core.Parameter(
        name="checkpoint",
        type=pathlib.Path,
        help="Load context from checkpoint.",
        allow_from_file=False
    ),
    core.Parameter(
        name="cuda",
        default=True,
        is_flag=True
    ),
    core.Parameter(
        name="distributed",
        help="Launch in distributed mode.",
        default=False,
        allow_from_file=False,
        is_flag=True
    ),
    core.Parameter(
        name="local_rank", 
        type=int,
        help="For distributed training, gives the local rank for this process. "
            "This parameter is set automatically by the framework, and should not "
            "be used manually.",
        allow_from_file=False,
        hidden=True,
    ),
    core.Parameter(
        name="nproc_per_node",
        envvar="NPROC_PER_NODE", 
        type=int, 
        default=1,
        help="The number of processes to launch on each node, "
            "for GPU training, this is recommended to be set "
            "to the number of GPUs in your system so that "
            "each process can be bound to a single GPU.",
    ),
    core.Parameter(
        name="node_rank", 
        envvar="NODE_RANK",
        type=int, 
        default=0,
        help="The rank of the node for multi-node distributed training.",
        allow_from_file=False,
    ),
    core.Parameter(
        name="nnodes", 
        envvar="NNODES",
        type=int, 
        default=1,
        help="The number of nodes to use for distributed training.",
    ),
    core.Parameter(
        name="master_addr", 
        default="127.0.0.1",
        envvar="MASTER_ADDR",
        type=str,
        help="Master node (rank 0)'s address, should be either "
            "the IP address or the hostname of node 0. Leave "
            "default for single node training.",
    ),
    core.Parameter(
        name="master_port", 
        default=29500, 
        envvar="MASTER_PORT",
        type=int,
        help="Master node (rank 0)'s free port that needs to "
            "be used for communciation during distributed "
            "training.",
    ),
]

api_object = TorchApi()
api_object.parameter = parameter_decorator
api_object.experiment = functools.partial(experiment_decorator, api_object=api_object, default_params=classic_params+pytorch_params)
api_object.group = functools.partial(group_decorator, experiment_decorator=api_object.experiment)
api_object.ContextObject = context.ContextObject
api_object.ContextDict = context.ContextDict
api_object.ContextList = context.ContextList
api_object.Recorder = recording.Recorder
api_object.BetterDataLoader = torchutil.BetterDataLoader

register_plugin(api_object, "torch")
