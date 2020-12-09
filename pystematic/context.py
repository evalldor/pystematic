import datetime
import json
import logging
import os
import pathlib
import random
import re
import subprocess
import sys
import dataclasses
import typing

import click
import numpy as np
import torch

from .torchutil import Logger

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

    def __init__(self, config, log_handlers=[BasicLogHandler()]):
        self._config = config
        self._output_dir = None

        # Log level
        if self.config("debug"):
            config.update({
                "log_level": "DEBUG"
            })

        logging.basicConfig(level=self.config("log_level"), handlers=log_handlers)

        # Random gen
        if "random_seed" not in config or config["random_seed"] is None:
            self._config["random_seed"] = random.getrandbits(32)

        self._random_gen = random.Random(self._config["random_seed"])

        if not self.config_file.exists():
            logger.info(f"Writing config file to '{self.config_file}'.")
            with self.config_file.open("w") as f:
                json.dump(self.config(), f, indent=4)

    def config(self, name=None, default=None):
        """Returns the value of config item with name `name`. If name is `None`
        the whole config dict is returned. If item `name` does not exist it
        returns the value of `default`.
        """
        if name is None:
            return self._config.copy()
        
        if name in self._config:
            return self._config[name]
        
        return default

    @property
    def config_file(self):
        return self.output_dir.joinpath("config.json")

    @property
    def output_dir(self):
        """Returns the path to the directory where all output files should be
        saved. The path is wrapped in a `pathlib.Path` object.
        """
        if self._output_dir is None:
            if self.config("internal_logdir") is not None:
                self._output_dir = pathlib.Path(self.config("internal_logdir"))
            else:
                self._output_dir = _create_log_dir_name(self.config("output_dir"), self.config("_experiment_name"))
                
                logger.info(f"Logdir is '{self._output_dir}'.")
                
                self._output_dir.mkdir(parents=True, exist_ok=True)

        return self._output_dir

    def new_seed(self, nbits=32):
        """Use this function to generate random numbers seeded by the config
        value `random_seed`. Expected use is to seed your own random number
        generators.
        """
        return self._random_gen.getrandbits(nbits)



def _launch_distributed_subprocesses(config, config_file, logdir):
    logger.info("Launching distributed subprocesses...")

    all_subprocesses = []
    for local_rank in range(config["nproc_per_node"]):
        
        cmd = [sys.executable] + sys.argv[:]
        cmd.append("--config-file")
        cmd.append(str(config_file))
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

def _config_dict_to_args(config):
    """Converts a dict to a cmd option string"""
    args = []

    for key, value in config.items():
        if value is not None:
            args.append(f"--{key.replace('_', '-')}")
            args.append(str(value))
    
    return args

def _normalize_state_dict(state_dict):
    """This removes any key prefixes inserted by DistributedDataParallel so that
    a model can be loaded without running in distributed mode
    """
    new_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = re.sub(r"^module\.", "", key)
            new_dict[new_key] = value
        else:
            new_dict[key] = value

    return new_dict

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
        level = click.style("[{}]".format(record.levelname), fg=self._colors[record.levelname])  
        msg = click.style("{}".format(record.getMessage()), fg="white")

        name = click.style("[{}]".format(record.name), fg="green")
        
        if torch.distributed.is_initialized():
            rank = click.style(f"[RANK {torch.distributed.get_rank()}]", fg="green")
            click.echo("{} {} {} {}".format(level, rank, name, msg))
        else:
            click.echo("{} {} {}".format(level, name, msg))


@dataclasses.dataclass
class ContextItem:
    """Represents an item registered with a pytorch context. It is just a
    wrapper around an object together with some configuration options."""
    
    handle: typing.Any # A handle to the actual item / object

    cuda: bool # Can be used to disable cuda movement for specific items
    checkpoint: bool # Set to False to exclude this item from being saved and loaded from checkpoints
    


class PytorchContext(BasicContext):

    def __init__(self, config):
        super().__init__(config, log_handlers=[PytorchLogHandler()])

        if self.config("distributed") and self.config("local_rank") is None:
            # Launch sub processes like in torch.distributed.launch
            _launch_distributed_subprocesses(self.config(), self.config_file, self.output_dir)
            exit(0)
        

        self._items = {
            "global_step": 0,
            "epoch": 0,
            "__torch_version": torch.__version__
        }

        self._checkpoint = {}


        if self.config("checkpoint") is not None:
            self.load_checkpoint(self.config("checkpoint"))

        if self.config("distributed") and self.config("local_rank") is not None:
            global_rank = self.config("nproc_per_node") * self.config("node_rank") + self.config("local_rank")
            world_size = self.config("nproc_per_node") * self.config("nnodes")

            logger.info(f"Initializing distributed runtime (world size '{world_size}', local rank '{self.config('local_rank')}', global rank '{global_rank}')...")
            
            torch.cuda.set_device(self.config("local_rank"))
            
            torch.distributed.init_process_group(
                backend='nccl',
                init_method="tcp://{}:{}".format(self.config("master_addr"), self.config("master_port")),
                world_size=world_size,
                rank=global_rank
            )

            logger.info(f"Distributed runtime initialized. World size is '{torch.distributed.get_world_size()}'.")
    
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
            return [self.config("local_rank")]
        
        return [i for i in range(torch.cuda.device_count())]

    def get_rank(self):
        if self.is_distributed():
            return torch.distributed.get_rank()
        
        return 0

    def all_reduce_sum(self, value):
        """Sums 'value' value across all processes"""
        value = torch.tensor(value)

        if self.is_distributed():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)

        return value
    
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
        
        if self.config("cuda") and callable(getattr(item, "cuda", None)):
            if cuda:
                logger.info(f"Moving '{name}' to CUDA.")
                item = item.cuda()
            else:
                logger.debug(f"Not moving item '{name}' to cuda due to item specific config.")
        
        
        if isinstance(item, torch.nn.Module) and not isinstance(item, torch.nn.modules.loss._Loss):
            if self.is_distributed():
                logger.info(f"Converting to distributed for model '{name}'.")
                
                item = torch.nn.parallel.DistributedDataParallel(
                    module=item,
                    device_ids=self.get_allocated_device_ids(),
                    output_device=self.get_allocated_device_ids()[0],
                    broadcast_buffers=broadcast_buffers
                )
        
        self._items[name] = ContextItem(handle=item, cuda=cuda, checkpoint=checkpoint)

    def has(self, name):
        return name in self._items.keys()
    
    def get(self, name):
        if name not in self._items.keys():
            #Add default logger if none exists
            if name.lower() == "logger":
                self._items["logger"] = Logger(str(self.output_dir), dummy=(not self.is_master()))
            else:
                raise Exception("Name '{}' not registered.".format(name))

        return self._items[name]

    #
    # Tasks
    #
    def step_global(self):
        """Increases the 'global_step' counter by 1."""
        self._items["global_step"] += 1

    def step_epoch(self):
        """Increases the 'epoch' counter by 1."""
        self._items["epoch"] += 1

    def place_on_correct_device(self, *args):
        """Utility method to place a batch of data on the correct device (i.e.
        cuda or cpu) depending on the 'cuda' config flag."""
        res = []
        for arg in args:
            if self.config("cuda") and callable(getattr(arg, "cuda", None)):
                res.append(arg.cuda())
            else:
                res.append(arg)
        return res
    
    def seed_random_generators(self):
        """This is just a helper to seed all known random modules with reproducible seeds."""
        self.seed_python_random_module()
        self.seed_pytorch_random_module()
        self.seed_numpy_random_module()

    def seed_python_random_module(self):
        seed = self.new_seed()
        logger.info(f"Seeding python random module with '{seed}'.")
        random.seed(seed)

    def seed_pytorch_random_module(self):
        seed = self.new_seed()
        logger.info(f"Seeding pytorch random module with '{seed}'.")
        torch.manual_seed(seed)

    def seed_numpy_random_module(self):
        seed = self.new_seed()
        logger.info(f"Seeding numpy random module with '{seed}'.")
        np.random.seed(seed)

    #
    # Checkpoint
    #
    def save_checkpoint(self):
        if self.is_master():
            checkpoint_file_path = self.output_dir.joinpath(f"checkpoint-{self.get('epoch')}.ckpt")

            logger.info(f"Saving checkpoint '{checkpoint_file_path}'.")

            dict_to_save = {
                "epoch": self.get("epoch"),
                "global_step": self.get("global_step"),
                "__torch_version": self.get("__torch_version")
            }

            for name, item in self._items.items():
                if callable(getattr(item.handle, "state_dict", None)):
                    if item.checkpoint:
                        dict_to_save[name] = _normalize_state_dict(item.handle.state_dict())
                    else:
                        logger.debug(f"Not saving item '{name}' to checkpoint due to item specific config.")

            with checkpoint_file_path.open("wb") as f:
                torch.save(dict_to_save, f)

    def load_checkpoint(self, checkpoint_file_path):
        """Loads the checkpoint from the given filepath. Note that this method
        does not actually populate the items with values, it just loads the
        checkpoint into an internal dict. The actual population of the loaded
        values is done when an item is registered with the context. It is
        therefore required that the checkpoint be loaded BEFORE any item is
        registered with the context."""

        logger.info(f"Loading checkpoint '{checkpoint_file_path}'.")

        with open(checkpoint_file_path, "rb") as f:
            self._checkpoint = torch.load(f, map_location="cpu")

        self._items["epoch"] = self._checkpoint["epoch"]
        self._items["global_step"] = self._checkpoint["global_step"]
