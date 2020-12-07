import functools
import inspect
import json
import logging
import random
import re
import os

import click

from .context import BasicContext, PytorchContext


logger = logging.getLogger("Cli")

class Option(click.Option):
    """Extends the default click option with a flag for whether the option is
    allowed to be loaded from config file.
    """

    def __init__(self, param_decls=None, allow_from_cfgfile=True, **attrs):
        super().__init__(param_decls, **attrs)

        self.allow_from_cfgfile = allow_from_cfgfile


class Label(click.Option):

    def __init__(self, name):
        super().__init__([], expose_value=False, hidden=True, allow_from_autoenv=False)
        self.label = name


class Experiment(click.Command):

    # def __init__(self, *args, **attrs):
    #     super().__init__(*args, **attrs)
    
    def format_options(self, ctx, formatter):
        """Writes all the options into the formatter if they exist. Allows for
        `Label` options.
        """

        opts = {
            None: []
        }
        labels = [None]
        current_label = None

        for param in self.get_params(ctx):

            if isinstance(param, Label):
                current_label = param.label
                labels.append(current_label)
                opts[current_label] = []
            else:
                rv = param.get_help_record(ctx)

                if rv is not None:
                    opts[current_label].append(rv)

        
        with formatter.section("Options"):   

            for label in labels:
                if len(opts[label]) > 0:
                    if label is not None:
                        with formatter.section(label):
                            formatter.write_dl(opts[label], col_max=40)
                    else:
                        formatter.write_dl(opts[label], col_max=40)


def _config_file_callback(ctx, param, file_path):

    if file_path is not None:
        blacklisted_config_ops = []

        for param in ctx.command.params:
            if hasattr(param, "allow_from_cfgfile") and not param.allow_from_cfgfile:
                blacklisted_config_ops.append(param.name)

        with open(file_path) as d:
            cfg = json.load(d)

        for key in blacklisted_config_ops:
            cfg.pop(key, None)

        ctx.default_map.update(cfg)


def _help_callback(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return

    click.echo(ctx.get_help(), color=ctx.color)
    ctx.exit()


def _continue_from_callback(ctx, param, dir_path):
    if dir_path is not None:
        cfg_file = _find_config_file_in_dir(dir_path)

        if cfg_file is not None:
            _config_file_callback(ctx, "config_file", cfg_file)
        else:
            logger.warn(f"Could not find a config file in directory '{dir_path}'.")
        
        ckpt_file = _find_latest_checkpoint_in_dir(dir_path)
        if ckpt_file is not None:
            ctx.default_map.update({
                "checkpoint": ckpt_file
            })
        else:
            logger.warn(f"Could not find a checkpoint in directory '{dir_path}'.")

def _find_config_file_in_dir(dir_path):
    config_path = os.path.join(dir_path, "config.json")
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

@click.group()
def experiments_main():
    pass


def _get_params(f):
    if isinstance(f, click.Command):
        return f.params
    elif hasattr(f, "__click_params__"):
        return f.__click_params__
    
    return []

general_options = [
    Label("General"),
    click.Option(["-h", "--help"], 
        is_flag=True,
        expose_value=False,
        is_eager=True,
        help="Show this message and exit.",
        callback=_help_callback
    ),
    click.Option(["--output-dir"],
        default="./output",
        help="Directory to store all run-logs. Will be created if it does not exist.",
        type=click.Path(file_okay=False),
        show_default=True,
        show_envvar=True
    ),
    click.Option(["--log-level"],
        type=click.Choice(["INFO", "DEBUG"], case_sensitive=False),
        default="INFO",
        help="Log level.",
        show_default=True,
        show_envvar=True
    ),
    click.Option(["--debug/--nodebug"],
        default=False,
        help="Sets debug flag on/off.",
        show_default=True,
        show_envvar=True
    ),
    Option(['--config-file'],
        type=click.Path(dir_okay=False),
        is_eager=True,
        help="Read configuration from FILE.",
        callback=_config_file_callback,
        allow_from_cfgfile=False
    ),
    click.Option(["--random-seed"],
        default=functools.partial(random.getrandbits, 32),
        help="The value to seed random number generators with.",
        type=int, 
        show_default="randomly generated"
    ),
    click.Option(["--comment"], 
        type=str,
        help="A string comment to add to the config file for reference.",
        show_default=True,
        show_envvar=True
    ),
    Option(["--internal-logdir"],
        default=None,
        help="Internally used to refer to an already created log dir. DO NOT USE MANUALLY.",
        type=click.Path(file_okay=False),
        allow_from_cfgfile=False,
        hidden=True
    )
]

def experiment(name=None, extends=None, defaults=None, **kwargs):

    kwargs["context_settings"] = {
        "default_map": defaults,
        "show_default": True
    }

    def decorator(func):
        @functools.wraps(func)
        def command_wrapper(**config):
            ctx = BasicContext(config)

            return func(ctx)

        cmd = click.decorators._make_command(command_wrapper, name, attrs=kwargs, cls=Experiment)
        
        if extends is not None:
            cmd.params += _get_params(extends)

        cmd.params += general_options

        experiments_main.add_command(cmd)

        return cmd

    if callable(name):
        func = name
        name = None
        return decorator(func)
    
    else:
        return decorator


pytorch_options = [
    Label("Training"),
    Option(["--checkpoint"],
        type=click.Path(dir_okay=False),
        help="Load context from checkpoint.",
        show_default=True,
        show_envvar=True,
        allow_from_cfgfile=False
    ),
    Option(["--continue-from"],
        type=click.Path(file_okay=False),
        is_eager=True,
        callback=_continue_from_callback,
        help="Continue from the latest checkpoint found in dir.",
        show_default=True,
        show_envvar=True,
        allow_from_cfgfile=False
    ),
    click.Option(["--cuda/--nocuda"],
        default=True,
        show_default=True,
        show_envvar=True
    ),
    click.Option(["--syncbn/--nosyncbn"], 
        default=False,
        help='APEX synchronized batch normalization.',
        show_default=True,
        show_envvar=True
    ),

    Label("Distributed"),
    click.Option(["--distributed/--nodistributed"],
        help="Launch in distributed mode",
        default=False,
        show_default=True,
        show_envvar=True
    ),
    Option(["--local-rank"], 
        type=int,
        help="For distributed training, gives the local rank for this process.",
        show_default=True,
        show_envvar=True,
        allow_from_cfgfile=False
    ),
    click.Option(["--nproc-per-node"],
        envvar="NPROC_PER_NODE", 
        type=int, 
        default=1,
        help="The number of processes to launch on each node, "
            "for GPU training, this is recommended to be set "
            "to the number of GPUs in your system so that "
            "each process can be bound to a single GPU.",
        show_default=True,
        show_envvar=True
    ),
    Option(["--node-rank"], 
        envvar="NODE_RANK",
        type=int, 
        default=0,
        help="The rank of the node for multi-node distributed training.",
        show_default=True,
        show_envvar=True,
        allow_from_cfgfile=False
    ),
    click.Option(["--nnodes"], 
        envvar="NNODES",
        type=int, 
        default=1,
        help="The number of nodes to use for distributed training.",
        show_default=True,
        show_envvar=True
    ),
    click.Option(["--master-addr"], 
        default="127.0.0.1",
        envvar="MASTER_ADDR",
        type=str,
        help="Master node (rank 0)'s address, should be either "
            "the IP address or the hostname of node 0. Leave "
            "default for single node training.",
        show_default=True,
        show_envvar=True
    ),
    click.Option(["--master-port"], 
        default=29500, 
        envvar="MASTER_PORT",
        type=int,
        help="Master node (rank 0)'s free port that needs to "
            "be used for communciation during distributed "
            "training."
    )
]

def pytorch_experiment(name=None, extends=None, defaults=None, **kwargs):

    kwargs["context_settings"] = {
        "default_map": defaults,
        "show_default": True
    }
        
    def decorator(func):
        @functools.wraps(func)
        def command_wrapper(**config):
            ctx = PytorchContext(config)

            return func(ctx)

        functools.update_wrapper(command_wrapper, func)

        cmd = click.decorators._make_command(command_wrapper, name, attrs=kwargs, cls=Experiment)

        if extends is not None:
            cmd.params += _get_params(extends)

        cmd.params += general_options + pytorch_options

        experiments_main.add_command(cmd)

        return cmd

    if callable(name):
        func = name
        name = None
        return decorator(func)
    
    else:
        return decorator
