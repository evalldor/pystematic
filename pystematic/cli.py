import functools
import inspect
import logging
import random
import re
import os
import pathlib

import yaml

import click

from . import torchapi

from .click_adapter import Parameter, Label, make_experiment_decorator

logger = logging.getLogger("cli")





def _params_file_callback(ctx, param, file_path):

    if file_path is not None:
        blacklisted_config_ops = []

        for param in ctx.command.params:
            if hasattr(param, "allow_from_params_file") and not param.allow_from_params_file:
                blacklisted_config_ops.append(param.name)
        
        file_path = pathlib.Path(file_path)

        if not file_path.exists():
            logger.warn(f"Could not find parameters file '{file_path}'.")
        else:
            with file_path.open("r") as f:
                params_file = yaml.load(f, Loader=yaml.Loader)
            
            for key in blacklisted_config_ops:
                params_file.pop(key, None)
       
        return str(file_path)


def _help_callback(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return

    click.echo(ctx.get_help(), color=ctx.color)
    ctx.exit()


def _continue_from_callback(ctx, param, dir_path):
    if dir_path is not None:
        cfg_file = _find_params_file_in_dir(dir_path)

        if cfg_file is not None:
            _params_file_callback(ctx, "params_file", cfg_file)
        else:
            logger.warn(f"Could not find a parameters file in directory '{dir_path}'.")
        
        ckpt_file = _find_latest_checkpoint_in_dir(dir_path)
        if ckpt_file is not None:
            ctx.default_map.update({
                "checkpoint": ckpt_file
            })
        else:
            logger.warn(f"Could not find a checkpoint in directory '{dir_path}'.")


def _find_params_file_in_dir(dir_path):
    config_path = os.path.join(dir_path, "parameters.yml")
    if os.path.exists(config_path):
        return config_path
    
    return None


def _find_latest_checkpoint_in_dir(dir_path):
    latest = -1

    for filename in os.listdir(dir_path):
        match = re.match(r"^checkpoint-([0-9]+).ckpt$", filename)
        if match:
            idx = int(match.group(1))
            if idx > latest:
                latest = idx

    if latest == -1:
        return None

    return os.path.join(dir_path, "checkpoint-{}.ckpt".format(latest))



general_options = [
    Label("General"),
    click.Option(["-h", "--help"], 
        is_flag=True,
        expose_value=False,
        is_eager=True,
        help="Show this message and exit.",
        callback=_help_callback
    ),
    Parameter(
        name="output_dir",
        default="./output",
        help=inspect.cleandoc(
            """Parent directory to store all run-logs in. Will be created if it
            does not exist."""
        ),
        type=click.Path(file_okay=False),
        show_default=True,
        show_envvar=True
    ),
    # click.Option(["--log-level"],
    #     type=click.Choice(["INFO", "DEBUG"], case_sensitive=False),
    #     default="INFO",
    #     help="Log level.",
    #     show_default=True,
    #     show_envvar=True
    # ),
    Parameter(
        name="debug",
        is_flag=True,
        default=False,
        help="Sets debug flag on/off.",
        show_default=True,
        show_envvar=True
    ),
    Parameter(
        name="params_file",
        type=click.Path(dir_okay=False),
        is_eager=True,
        help="Read experiment parameters from FILE.",
        callback=_params_file_callback,
        allow_from_params_file=False
    ),
    Parameter(
        name="random_seed",
        default=functools.partial(random.getrandbits, 32),
        help="The value to seed random number generators with.",
        type=int, 
        show_default="randomly generated"
    ),
    Parameter(
        name="subprocess",
        default=None,
        help="Internally used to indicate that this process is a subprocess. DO NOT USE MANUALLY.",
        is_eager=True,
        callback=_params_file_callback,
        type=click.Path(dir_okay=False),
        allow_from_params_file=False,
        hidden=True
    )
]

pytorch_options = [
    Label("Training"),
    Parameter(
        name="checkpoint",
        type=click.Path(dir_okay=False),
        help="Load context from checkpoint.",
        allow_from_params_file=False
    ),
    # Parameter(
    #     name="continue_from",
    #     type=click.Path(file_okay=False),
    #     is_eager=True,
    #     callback=_continue_from_callback,
    #     help="Continue from the latest checkpoint found in dir.",
    #     show_default=True,
    #     show_envvar=True,
    #     allow_from_params_file=False
    # ),
    click.Option(["--cuda/--nocuda"],
        default=True,
    ),
    
    Label("Distributed"),
    Parameter(
        name="distributed",
        help="Launch in distributed mode",
        is_flag=True,
        default=False,
        allow_from_params_file=False
    ),
    Parameter(
        name="local_rank", 
        type=int,
        help="For distributed training, gives the local rank for this process.",
        allow_from_params_file=False
    ),
    Parameter(
        name="nproc_per_node",
        envvar="NPROC_PER_NODE", 
        type=int, 
        default=1,
        help="The number of processes to launch on each node, "
            "for GPU training, this is recommended to be set "
            "to the number of GPUs in your system so that "
            "each process can be bound to a single GPU.",
    ),
    Parameter(
        name="node_rank", 
        envvar="NODE_RANK",
        type=int, 
        default=0,
        help="The rank of the node for multi-node distributed training.",
        allow_from_params_file=False
    ),
    Parameter(
        name="nnodes", 
        envvar="NNODES",
        type=int, 
        default=1,
        help="The number of nodes to use for distributed training.",
    ),
    Parameter(
        name="master_addr", 
        default="127.0.0.1",
        envvar="MASTER_ADDR",
        type=str,
        help="Master node (rank 0)'s address, should be either "
            "the IP address or the hostname of node 0. Leave "
            "default for single node training.",
    ),
    Parameter(
        name="master_port", 
        default=29500, 
        envvar="MASTER_PORT",
        type=int,
        help="Master node (rank 0)'s free port that needs to "
            "be used for communciation during distributed "
            "training."
    )
]


# experiment = make_experiment_decorator(general_options, BasicContext)

def pytorch_experiment_initializer(experiment_main, params):
    
    torchapi._initialize(params)

    return experiment_main(params, torchapi.context)

pytorch_experiment = make_experiment_decorator(general_options + pytorch_options, pytorch_experiment_initializer)
