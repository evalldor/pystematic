import datetime
import logging
import os
import pathlib
import random
import re
import subprocess
import sys
import dataclasses
import typing

import yaml

import click
import numpy as np
import torch

from .recording import Recorder

logger = logging.getLogger('Context')

def _create_log_dir_name(output_dir, experiment_name):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return pathlib.Path(output_dir).resolve().joinpath(experiment_name).joinpath(current_time)


class BasicLogHandler(logging.Handler):

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

        click.echo(f"{level} {name} {msg}")



class BasicContext:

    def __init__(self, params):

        if params["debug"]:
            params.update({
                "log_level": "DEBUG"
            })

        logging.basicConfig(level=params["log_level"], handlers=[BasicLogHandler()])
        
        if "random_seed" not in params or params["random_seed"] is None:
            params["random_seed"] = random.getrandbits(32)

        if params["internal_logdir"] is not None:
            output_dir = pathlib.Path(params["internal_logdir"])
        else:
            output_dir = _create_log_dir_name(params["output_dir"], params["_experiment_name"])
            output_dir.mkdir(parents=True, exist_ok=True)

        params_file = output_dir.joinpath("parameters.yml")

        logger.info(f"Output dir is '{output_dir}'.")

        if not params_file.exists():
            logger.info(f"Writing parameters file to '{output_dir}'.")
            with params_file.open("w") as f:
                yaml.dump(params, f, default_flow_style=False)

        self._params = params
        self._random_generator = random.Random(params["random_seed"])
        self._output_dir = output_dir

    def param(self, name : str = None):
        """Returns the value of the parameter with name `name`. If name is `None`
        the whole parameters dict is returned.
        """
        if name is None:
            return self._params.copy()
        
        if name in self._params:
            return self._params[name]
        
        raise f"Parameter with name '{name}' does not exist!"

    @property
    def params_file(self) -> pathlib.Path:
        return self._output_dir.joinpath("parameters.yml")

    @property
    def output_dir(self) -> pathlib.Path:
        """Returns the path to the directory where all output files should be
        saved. The path is wrapped in a `pathlib.Path` object.
        """
        return self._output_dir

    def new_seed(self, nbits=32) -> int:
        """Use this function to generate random numbers seeded by the experiment
        parameter `random_seed`. Expected use is to seed your own random number
        generators.
        """
        return self._random_generator.getrandbits(nbits)



def _launch_distributed_subprocesses(params, params_file, logdir):
    logger.info("Launching distributed subprocesses...")

    all_subprocesses = []
    for local_rank in range(params["nproc_per_node"]):
        
        cmd = [sys.executable] + sys.argv[:]
        cmd.append("--params-file")
        cmd.append(str(params_file))
        cmd.append("--internal-logdir")
        cmd.append(str(logdir))
        cmd.append("--local-rank")
        cmd.append(str(local_rank))
        
        logger.debug(f"Executing '{' '.join(cmd)}'.")

        process = subprocess.Popen(cmd, env=os.environ.copy())
        all_subprocesses.append(process)
    
    logger.info(f"Launched '{len(all_subprocesses)}' subprocesses. Waiting for termination...")
    
    for process in all_subprocesses:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=sys.argv[:])

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


@dataclasses.dataclass
class ContextItem:
    """Represents an item registered with a pytorch context. It is just a
    wrapper around an object together with some configuration items."""
    
    handle: typing.Any # A handle to the actual item / object

    cuda: bool = True # Can be used to disable cuda movement for specific items
    checkpoint: bool = True # Set to False to exclude this item from being saved and loaded from checkpoints


class PytorchContext(BasicContext):

    def __init__(self, params):
        super().__init__(params)
        
        logging.basicConfig(level=self.param("log_level"), handlers=[PytorchLogHandler()])

        if self.param("distributed") and self.param("local_rank") is None:
            # Launch sub processes like in torch.distributed.launch
            _launch_distributed_subprocesses(self.param(), self.params_file, self.output_dir)
            exit(0)
        
        if self.param("distributed") and self.param("local_rank") is not None:
            global_rank = self.param("nproc_per_node") * self.param("node_rank") + self.param("local_rank")
            world_size = self.param("nproc_per_node") * self.param("nnodes")

            logger.info(f"Initializing distributed runtime (world size '{world_size}', local rank '{self.param('local_rank')}', global rank '{global_rank}')...")
            
            torch.cuda.set_device(self.param("local_rank"))
            
            torch.distributed.init_process_group(
                backend='nccl',
                init_method="tcp://{}:{}".format(self.param("master_addr"), self.param("master_port")),
                world_size=world_size,
                rank=global_rank
            )

            logger.info(f"Distributed runtime initialized. World size is '{torch.distributed.get_world_size()}'.")

        self._checkpoint = {}
        self._items = {}

        if self.param("checkpoint") is not None:
            self.load_checkpoint(self.param("checkpoint"))

    #
    # Distributed
    #
    
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
            return [self.param("local_rank")]
        
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
    # Items/Components
    #

    def add(self, name, item, cuda=True, checkpoint=True, broadcast_buffers=True):
        if name in self._items.keys():
            raise Exception("Names need to be unique. Name '{}' already registered.".format(name))

        if name in self._checkpoint:
            if checkpoint:
                logger.info(f"Loading parameters from checkpoint for '{name}'.")
                item.load_state_dict(self._checkpoint[name])
            else:
                logger.debug(f"Not loading item '{name}' from checkpoint due to item specific config.")
        
        if self.param("cuda") and callable(getattr(item, "cuda", None)):
            if cuda:
                logger.info(f"Moving '{name}' to CUDA.")
                item = item.cuda()
            else:
                logger.debug(f"Not moving item '{name}' to cuda due to item specific config.")
        
        
        if isinstance(item, torch.nn.Module) and len([*item.parameters()]) > 0:
            if self.is_distributed():
                logger.info(f"Converting to distributed for model '{name}'.")
                
                item = torch.nn.parallel.DistributedDataParallel(
                    module=item,
                    device_ids=self.get_allocated_device_ids(),
                    output_device=self.get_allocated_device_ids()[0],
                    broadcast_buffers=broadcast_buffers
                )

        elif isinstance(item, Recorder):
            item.set_output_dir(self.output_dir)

            if not self.is_master():
                item.silence() # We silence all recorders which do not belong to the master process
        
        self._items[name] = ContextItem(handle=item, cuda=cuda, checkpoint=checkpoint)

    def has(self, name):
        return name in self._items.keys()
    
    def get(self, name):
        if name not in self._items.keys():
            raise Exception("Name '{}' not registered.".format(name))

        return self._items[name].handle

    def delete(self, name):
        if name in self._items:
            del self._items[name]

    #
    # Tasks
    #

    def place_on_correct_device(self, *args):
        """Utility method to place a batch of data on the correct device (i.e.
        cuda or cpu) depending on the 'cuda' experiment parameter."""
        res = []
        for arg in args:
            if self.param("cuda") and callable(getattr(arg, "cuda", None)):
                res.append(arg.cuda())
            else:
                res.append(arg)
        return res
    
    def seed_random_generators(self):
        """This is just a helper to seed all known random modules with reproducible seeds."""
        
        logger.info(f"Seeding random modules for python, numpy and pytorch.")

        random.seed(self.new_seed())
        torch.manual_seed(self.new_seed())
        np.random.seed(self.new_seed())


    #
    # Checkpoint
    #

    def save_checkpoint(self, suffix):
        """Saves registered items to a file. All items that have a function
        named 'state_dict' will be saved by calling that function and saving the
        returned value.

        Args: suffix (str): Will be appended to the checkpoint filename.
        """

        if self.is_master():
            checkpoint_file_path = self.output_dir.joinpath(f"checkpoint-{suffix}.ckpt")

            logger.info(f"Saving checkpoint '{checkpoint_file_path}'.")

            dict_to_save = {
                "__torch_version": torch.__version__,
            }

            for name, item in self._items.items():
                if callable(getattr(item.handle, "state_dict", None)):
                    if item.checkpoint:
                        if isinstance(item.handle, torch.nn.parallel.DistributedDataParallel):
                            dict_to_save[name] = item.handle.module.state_dict()
                        else:
                            dict_to_save[name] = item.handle.state_dict()
                    else:
                        logger.debug(f"Not saving item '{name}' to checkpoint due to item specific config.")

            with checkpoint_file_path.open("wb") as f:
                torch.save(dict_to_save, f)

    def load_checkpoint(self, checkpoint_file_path):
        """Loads a checkpoint from the given filepath. Note that this method
        does not actually populate the items with values, it just loads the
        checkpoint into an internal dict. The actual population of the loaded
        values is done when an item is registered with the context. It is
        therefore required that the checkpoint be loaded BEFORE any item is
        registered with the context."""

        logger.info(f"Loading checkpoint '{checkpoint_file_path}'.")

        with open(checkpoint_file_path, "rb") as f:
            self._checkpoint = torch.load(f, map_location="cpu")
        
