import functools
import inspect
import logging
import random
import re
import os
import pathlib

import yaml

import click

from .context import BasicContext, PytorchContext


logger = logging.getLogger("Cli")


def _param_memo(f, param):
    if isinstance(f, click.Command):
        f.params.append(param)
    else:
        if not hasattr(f, "__click_params__"):
            f.__click_params__ = []
        f.__click_params__.append(param)

class Parameter(click.Option):
    """Extends the default click option with a flag for whether the option is
    allowed to be loaded from a parameters file.
    """

    def __init__(self, *, name, allow_from_params_file=True, **attrs):

        assert not "-" in name, "name cannot contain '-' characters. Use '_' instead!"

        param_decls = f"--{name.lower().replace('_', '-')}"

        super().__init__([param_decls], **attrs)

        self.allow_from_params_file = allow_from_params_file



def parameter_decorator(**attrs):

    def decorator(func):
        _param_memo(func, Parameter(**attrs))
        return func

    return decorator

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
                params_file = yaml.safe_load(f)

            for key in blacklisted_config_ops:
                params_file.pop(key, None)

            ctx.default_map.update(params_file)


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


def _get_attached_click_parameters(f):
    if isinstance(f, click.Command):
        return f.params
    elif hasattr(f, "__click_params__"):
        return f.__click_params__
    
    return []


@click.group()
def global_entrypoint():
    """pystematic global entrypoint. Below is a list of all registered
    experiments. Append the name of the experiment you would like to run to the
    commandline you invoked to run this script.
    """
    """All experiments are registered with this Click group. In your main
    script, simply call this function to access the CLI for all registered
    experiments.
    """
    pass


def make_experiment_decorator(options, context):

    def experiment_constructor(func=None, *, name=None, inherit_params=None, defaults=None, **kwargs):

        kwargs["context_settings"] = { # These are passed to the click Command class,
            "default_map": defaults, 
            "show_default": True
        }

        def decorator(func):
            
            experiment_name = name or func.__name__.lower().replace("_", "-")

            @functools.wraps(func)
            def command_wrapper(**config):
                
                config["_experiment_name"] = experiment_name
                
                ctx = context(config)

                return func(ctx)

            cmd = click.decorators._make_command(command_wrapper, experiment_name, attrs=kwargs, cls=Experiment)
            
            if inherit_params is not None:
                cmd.params += _get_attached_click_parameters(inherit_params)

            cmd.params += options

            # Register the command with the global entrypoint.
            global_entrypoint.add_command(cmd)

            return cmd

        if callable(func):
            return decorator(func)
        else:
            return decorator

    return experiment_constructor


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
        help=inspect.cleandoc(
            """Parent directory to store all run-logs in. Will be created if it
            does not exist."""
        ),
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
    Parameter(
        name="params_file",
        type=click.Path(dir_okay=False),
        is_eager=True,
        help="Read experiment parameters from FILE.",
        callback=_params_file_callback,
        allow_from_params_file=False
    ),
    click.Option(["--random-seed"],
        default=functools.partial(random.getrandbits, 32),
        help="The value to seed random number generators with.",
        type=int, 
        show_default="randomly generated"
    ),
    click.Option(["--comment"], 
        type=str,
        help="A string comment to add to the parameters file for reference.",
        show_default=True,
        show_envvar=True
    ),
    Parameter(
        name="internal_logdir",
        default=None,
        help="Internally used to refer to an already created log dir. DO NOT USE MANUALLY.",
        type=click.Path(file_okay=False),
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
        show_default=True,
        show_envvar=True,
        allow_from_params_file=False
    ),
    Parameter(
        name="continue_from",
        type=click.Path(file_okay=False),
        is_eager=True,
        callback=_continue_from_callback,
        help="Continue from the latest checkpoint found in dir.",
        show_default=True,
        show_envvar=True,
        allow_from_params_file=False
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
    Parameter(
        name="local_rank", 
        type=int,
        help="For distributed training, gives the local rank for this process.",
        show_default=True,
        show_envvar=True,
        allow_from_params_file=False
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
    Parameter(
        name="node_rank", 
        envvar="NODE_RANK",
        type=int, 
        default=0,
        help="The rank of the node for multi-node distributed training.",
        show_default=True,
        show_envvar=True,
        allow_from_params_file=False
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


# experiment = make_experiment_decorator(general_options, BasicContext)

pytorch_experiment = make_experiment_decorator(general_options + pytorch_options, PytorchContext)
