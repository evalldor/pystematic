import typing
import sys
import pathlib
import inspect
import random
import functools

from . import yaml_wrapper as yaml

from . import parametric



def parameter_decorator(
    name: str,
    type: typing.Callable[[str], typing.Any] = str,
    
    default: typing.Union[typing.Any, typing.Callable[[], typing.Any], None] = None,
    required: bool = False,
    allowed_values: list[typing.Any] = None,
    is_flag: bool = False,
    multiple: bool = False,
    allow_from_file: bool = True,
    envvar: typing.Union[str, None, typing.Literal[False]] = None,

    help: typing.Optional[str] = None,
    default_help: typing.Optional[str] = None,
):
    """Adds a parameter to an experiment.

    Args:
        name (str): The name of the parameter. The name must be a valid python identifier
        type (typing.Callable[[str], typing.Any], optional): The type of the parameter. Defaults to str.
        default (typing.Union[typing.Any, typing.Callable[[], typing.Any], None], optional): The default value of 
            the parameter. Can be either a value or a callable. Defaults to None.
        required (bool, optional): Set to True if this parameter is required. Defaults to False.
        allowed_values (list[typing.Any], optional): If given, the value must be in the list of allowed values. 
            Defaults to None.
        is_flag (bool, optional): When set to True, this parameter is assumed 
            to be a boolean flag. A flag parameter does not need to be given a 
            value on the command line. Its mere presence on the command line will 
            automatically assign it the value True. Defaults to False.
        multiple (bool, optional): When set to True, the parameter may appear 
            many times on the command line. It's value will be a list of values 
            given. Defaults to False.
        allow_from_file (bool, optional): Controls whether it should be allowed to load a value for this 
            parameter from a params file. Defaults to True.
        envvar (typing.Union[str, None, typing.Literal[False]], optional): Name of the environment variable. 
            Defaults to None.
        help (typing.Optional[str], optional): A help text for the parameter that will be 
            shown on the command line. Defaults to None.
        default_help (typing.Optional[str], optional): A help text for the default value. If None, the default 
            help text will be created by calling ``str(default_value)``. Defaults to None.
    """
    
    def decorator(experiment):
        behaviours = [PystematicParameterBehaviour()]
        nargs = None
        _type = type
        if is_flag:
            if allowed_values is not None:
                raise ValueError(f"Error in parameter declaration for '{name}': 'is_flag' is incompatible with 'allowed_values'.")
            
            if multiple:
                raise ValueError(f"Error in parameter declaration for '{name}': 'is_flag' is incompatible with 'multiple'.")
            
            behaviours.append(parametric.BooleanFlagBehaviour())
        else:
            if allowed_values is not None:
                _type = parametric.ChoiceType(allowed_values)
            elif _type == bool:
                _type = parametric.BooleanType()

        if multiple:
            nargs = "*"

        param = parametric.Parameter(
            name=name,
            type=_type,

            required=required,
            default=default,
            nargs=nargs,
            envvar=envvar,

            help=help,
            default_help=default_help,

            behaviour=parametric.CompositBehaviour(*behaviours),

            allow_from_file=allow_from_file
        )

        if isinstance(experiment, Experiment):
            experiment.add_parameter(param)
        else:
            if not hasattr(experiment, "__params_memo__"):
                experiment.__params_memo__ = []
            
            experiment.__params_memo__.append(param)

        return experiment

    return decorator


def experiment_decorator(name=None, inherit_params=None, defaults={}, group=None):
    if callable(name):
        main_function = name
        name = None
    else:
        main_function = None

    def decorator(main_function):
        experiment = Experiment(main_function, name=name, defaults_override=defaults)

        if inherit_params is not None:
            if not isinstance(inherit_params, (tuple, list)):
                experiments_to_inherit_from = [inherit_params]
            else:
                experiments_to_inherit_from = inherit_params

            for exp in experiments_to_inherit_from:
                if isinstance(exp, Experiment):
                    for param in exp.param_manager.get_parameters():
                        experiment.add_parameter(param)
                elif callable(exp):
                    if hasattr(exp, "__params_memo__"):
                        for param in exp.__params_memo__:
                            experiment.add_parameter(param)
                else:
                    raise ValueError(f"Unknown value passed to 'inherit_params': {exp}")

        if group is not None:
            group.add_experiment(experiment)

        return experiment

    if main_function:
        return decorator(main_function)
    
    return decorator


def experiment_group_decorator(name=None):
    if callable(name):
        main_function = name
        name = None
    else:
        main_function = None

    def decorator(main_function):
        group = ExperimentGroup(main_function, name=name)


        return group

    if main_function:
        return decorator(main_function)
    
    return decorator


class Experiment:

    def __init__(self, main_function, name=None, defaults_override={}):
        self.main_function = main_function
        self.name = name or main_function.__name__.lower().replace("_", "-")
        
        self.param_manager = parametric.ParameterManager(
            defaults_override=defaults_override,
            add_cli_help_option=True
        )

        for param in general_params + pytorch_params:
            self.add_parameter(param)

        if hasattr(main_function, "__params_memo__"):
            for param in main_function.__params_memo__:
                self.add_parameter(param)

    def add_parameter(self, param):
        self.param_manager.add_parameter(param)

    def get_parameters(self):
        return self.param_manager.get_parameters()

    def __call__(self, params):
        return self.run(params)

    def run(self, params):
        param_values = self.param_manager.from_dict(params)
        self._run_experiment(param_values)

    def cli(self, argv=None):
        if argv is None:
            argv = sys.argv[1:]

        param_values = self.param_manager.from_cli(argv)
        self._run_experiment(param_values)
    
    def _run_experiment(self, params):
        
        from . import torchapi
        global _current_experiment
        _current_experiment = self
        torchapi._initialize(params)
        self.main_function(params)


class ExperimentGroup:

    def __init__(self, main_function, name=None):
        self.main_function = main_function
        self.name = name or main_function.__name__.lower().replace("_", "-")

        self.experiments = []

        self.param_manager = parametric.ParameterManager(
            add_cli_help_option=True
        )

        self.param_manager.add_param(
            name="experiment",
            help="The name of the experiment to run.",
            required=True,
            cli_positional=True
        )

    def experiment(self, *args, **kwargs):
        kwargs["group"] = self
        return experiment_decorator(*args, **kwargs)

    def add_experiment(self, experiment):
        self.experiments.append(experiment)

    def cli(self, argv=None):
        if argv is None:
            argv = sys.argv[1:]

        param_values, argv_rest = self.param_manager.from_shared_cli(argv)
        
        experiments = {exp.name: exp for exp in self.experiments}

        exp_name = param_values["experiment"]
        if exp_name not in experiments:
            raise Exception(f"Invalid experiment name '{exp_name}'.")

        experiments[exp_name].cli(argv_rest)


class PystematicParameterBehaviour(parametric.DefaultParameterBehaviour):

    def after_init(self, param, allow_from_file=None, **kwargs):
        super().after_init(param, **kwargs)
        param.allow_from_file = allow_from_file


class ParamsFileBehaviour(parametric.DefaultParameterBehaviour):
    
    def on_value(self, param, value: pathlib.Path, result_dict: dict):
        super().on_value(param, value, result_dict)

        if value is not None:
            if not value.exists():
                raise ValueError(f"File does not exist: '{value}'.")

            blacklisted_config_ops = []

            for param in result_dict.get_params():
                if hasattr(param, "allow_from_file") and not param.allow_from_file:
                    blacklisted_config_ops.append(param.name)
            
            with value.open("r") as f:
                params_from_file = yaml.load(f)
            
            for key, value in params_from_file.items():
                if key not in blacklisted_config_ops:
                    result_dict.set_value_by_name(key, value)


general_params = [
    parametric.Parameter(
        name="output_dir",
        default="./output",
        help=inspect.cleandoc(
            """Parent directory to store all run-logs in. Will be created if it
            does not exist."""
        ),
        type=pathlib.Path,
        behaviour=PystematicParameterBehaviour()
    ),
    parametric.Parameter(
        name="debug",
        default=False,
        help="Sets debug flag on/off.",
        behaviour=parametric.CompositBehaviour(PystematicParameterBehaviour(), parametric.BooleanFlagBehaviour())
    ),
    parametric.Parameter(
        name="params_file",
        type=pathlib.Path,
        help="Read experiment parameters from FILE.",
        behaviour=parametric.CompositBehaviour(ParamsFileBehaviour(), PystematicParameterBehaviour()),
        allow_from_file=False
    ),
    parametric.Parameter(
        name="random_seed",
        default=functools.partial(random.getrandbits, 32),
        help="The value to seed random number generators with.",
        type=int, 
        default_help="randomly generated",
        behaviour=PystematicParameterBehaviour()
    ),
    parametric.Parameter(
        name="subprocess",
        default=None,
        help="Internally used to indicate that this process is a subprocess. DO NOT USE MANUALLY.",
        behaviour=parametric.CompositBehaviour(ParamsFileBehaviour(), PystematicParameterBehaviour()),
        type=pathlib.Path,
        allow_from_file=False,
        hidden=True
    )
]


pytorch_params = [
    parametric.Parameter(
        name="checkpoint",
        type=pathlib.Path,
        help="Load context from checkpoint.",
        allow_from_file=False,
        behaviour=PystematicParameterBehaviour()
    ),
    # parametric.Parameter(
    #     name="continue_from",
    #     type=click.Path(file_okay=False),
    #     is_eager=True,
    #     callback=_continue_from_callback,
    #     help="Continue from the latest checkpoint found in dir.",
    #     show_default=True,
    #     show_envvar=True,
    #     allow_from_file=False
    # ),
    parametric.Parameter(
        name="cuda",
        default=True,
        behaviour=parametric.CompositBehaviour(PystematicParameterBehaviour(), parametric.BooleanFlagBehaviour())
    ),
    
    parametric.Parameter(
        name="distributed",
        help="Launch in distributed mode.",
        default=False,
        allow_from_file=False,
        behaviour=parametric.CompositBehaviour(PystematicParameterBehaviour(), parametric.BooleanFlagBehaviour())
    ),
    parametric.Parameter(
        name="local_rank", 
        type=int,
        help="For distributed training, gives the local rank for this process. "
             "This parameter is set automatically by the framework, and should not "
             "be used manually.",
        allow_from_file=False,
        hidden=True,
        behaviour=PystematicParameterBehaviour()
    ),
    parametric.Parameter(
        name="nproc_per_node",
        envvar="NPROC_PER_NODE", 
        type=int, 
        default=1,
        help="The number of processes to launch on each node, "
            "for GPU training, this is recommended to be set "
            "to the number of GPUs in your system so that "
            "each process can be bound to a single GPU.",
        behaviour=PystematicParameterBehaviour()
    ),
    parametric.Parameter(
        name="node_rank", 
        envvar="NODE_RANK",
        type=int, 
        default=0,
        help="The rank of the node for multi-node distributed training.",
        allow_from_file=False,
        behaviour=PystematicParameterBehaviour()
    ),
    parametric.Parameter(
        name="nnodes", 
        envvar="NNODES",
        type=int, 
        default=1,
        help="The number of nodes to use for distributed training.",
        behaviour=PystematicParameterBehaviour()
    ),
    parametric.Parameter(
        name="master_addr", 
        default="127.0.0.1",
        envvar="MASTER_ADDR",
        type=str,
        help="Master node (rank 0)'s address, should be either "
            "the IP address or the hostname of node 0. Leave "
            "default for single node training.",
        behaviour=PystematicParameterBehaviour()
    ),
    parametric.Parameter(
        name="master_port", 
        default=29500, 
        envvar="MASTER_PORT",
        type=int,
        help="Master node (rank 0)'s free port that needs to "
            "be used for communciation during distributed "
            "training.",
        behaviour=PystematicParameterBehaviour()
    )
]


_current_experiment = None
def get_current_experiment():
    return _current_experiment
