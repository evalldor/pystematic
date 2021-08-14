import sys
import functools
import multiprocessing
import importlib
import typing
import importlib.metadata
import logging

from . import parametric

logger = logging.getLogger("pystematic.core")

class PystematicApp:

    def __init__(self) -> None:
        self._loaded_plugins = []

        self._experiment_created_callbacks = []
        self._before_experiment_callbacks = []
        self._after_experiment_callbacks = []
        
    def get_api_object(self):
        import pystematic
        return pystematic

    def load_all_plugins(self):
        # from .standard_plugin import StandardPlugin      
        # self._loaded_plugins.append(StandardPlugin(self))

        all_entry_points = importlib.metadata.entry_points()

        if "pystematic.plugins" in all_entry_points:
            
            for entry_point in all_entry_points["pystematic.plugins"]:
                logger.info(f"Loading plugin '{entry_point.name}'.")
                plugin = entry_point.load()
                self._loaded_plugins.append(plugin(self))

    def on_experiment_created(self, callback):
        self._experiment_created_callbacks.append(callback)

    def on_before_experiment(self, callback):
        self._before_experiment_callbacks.append(callback)

    def on_after_experiment(self, callback):
        self._after_experiment_callbacks.append(callback)


    def experiment_created(self, experiment):
        for callback in self._experiment_created_callbacks:
            experiment = callback(experiment)
        
        return experiment

    def before_experiment(self, experiment, params):
        for callback in self._before_experiment_callbacks:
            callback(experiment, params)

    def after_experiment(self):
        for callback in self._after_experiment_callbacks:
            callback()


app = PystematicApp()


class PystematicParameterBehaviour(parametric.DefaultParameterBehaviour):

    def after_init(self, param, allow_from_file=None, **kwargs):
        super().after_init(param, **kwargs)
        param.allow_from_file = allow_from_file


def Parameter(
    name: str,
    type: typing.Callable[[str], typing.Any] = str,
    
    default: typing.Union[typing.Any, typing.Callable[[], typing.Any], None] = None,
    required: bool = False,
    allowed_values: typing.List[typing.Any] = None,
    is_flag: bool = False,
    multiple: bool = False,
    allow_from_file: bool = True,
    envvar: typing.Union[str, None, typing.Literal[False]] = None,

    help: typing.Optional[str] = None,
    default_help: typing.Optional[str] = None,
    hidden = False,
    behaviour = None,
):
    behaviours = [PystematicParameterBehaviour()]

    if behaviour is not None:
        behaviours.append(behaviour)

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

    return parametric.Parameter(
        name=name,
        type=_type,

        required=required,
        default=default,
        nargs=nargs,
        envvar=envvar,

        help=help,
        default_help=default_help,
        hidden=hidden,
        behaviour=parametric.CompositBehaviour(*behaviours),

        allow_from_file=allow_from_file
    )


class Experiment:

    def __init__(self, main_function, name=None, defaults_override={}, no_output_dir=False):
        self.main_function = main_function
        self.name = name or main_function.__name__.lower().replace("_", "-")
        self.no_output_dir = no_output_dir
        self._defaults_override = defaults_override
        
        self.param_manager = parametric.ParameterManager(
            defaults_override=defaults_override,
            add_cli_help_option=True
        )

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

    def run_in_new_process(self, params):
        # We run the experiment like this to avoid pickling problems
        module = self.main_function.__module__
        name = self.main_function.__name__
        proc = multiprocessing.get_context('spawn').Process(
            target=_run_experiment_by_name,
            args=(module, name, params)
        )
        proc.start()

        return proc
    
    def _run_experiment(self, params):
        try:
            app.before_experiment(self, params)
            self.main_function(params)
        finally:
            app.after_experiment()


def _run_experiment_by_name(experiment_module, experiment_name, params):
    # used by Experiment.run_in_new_process
    module = importlib.import_module(experiment_module)
    getattr(module, experiment_name).run(params)


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

        self.experiment = functools.partial(experiment_decorator, group=self)

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


def parameter_decorator(
    name: str,
    type: typing.Callable[[str], typing.Any] = str,
    
    default: typing.Union[typing.Any, typing.Callable[[], typing.Any], None] = None,
    required: bool = False,
    allowed_values: typing.List[typing.Any] = None,
    is_flag: bool = False,
    multiple: bool = False,
    allow_from_file: bool = True,
    envvar: typing.Union[str, None, typing.Literal[False]] = None,

    help: typing.Optional[str] = None,
    default_help: typing.Optional[str] = None,
    hidden = False,
    behaviour = None,
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

        param = Parameter(
            name=name,
            type=type,
            
            default=default,
            required=required,
            allowed_values=allowed_values,
            is_flag=is_flag,
            multiple=multiple,
            allow_from_file=allow_from_file,
            envvar=envvar,

            help=help,
            default_help=default_help,
            hidden=hidden,
            behaviour=behaviour,
        )

        if isinstance(experiment, Experiment):
            experiment.add_parameter(param)
        else:
            if not hasattr(experiment, "__params_memo__"):
                experiment.__params_memo__ = []
            
            experiment.__params_memo__.append(param)

        return experiment

    return decorator


def experiment_decorator(
    name=None, 
    inherit_params=None, 
    defaults={}, 
    group=None,
    no_output_dir=False
):
    if callable(name):
        main_function = name
        name = None
    else:
        main_function = None

    def decorator(main_function):
        experiment = Experiment(
            main_function=main_function, 
            name=name, 
            defaults_override=defaults,
            no_output_dir=no_output_dir
        )

        experiment = app.experiment_created(experiment)

        if hasattr(main_function, "__params_memo__"):
            for param in main_function.__params_memo__:
                experiment.add_parameter(param)

        existing_params = [param.name for param in experiment.get_parameters()]

        if inherit_params is not None:
            if not isinstance(inherit_params, (tuple, list)):
                experiments_to_inherit_from = [inherit_params]
            else:
                experiments_to_inherit_from = inherit_params

            for exp in experiments_to_inherit_from:
                if isinstance(exp, Experiment):
                    for param in exp.param_manager.get_parameters():
                        if param.name not in existing_params:
                            experiment.add_parameter(param)
                elif callable(exp):
                    if hasattr(exp, "__params_memo__"):
                        for param in exp.__params_memo__:
                            if param.name not in existing_params:
                                experiment.add_parameter(param)
                else:
                    raise ValueError(f"Unknown value passed to 'inherit_params': {exp}")

        if group is not None:
            group.add_experiment(experiment)

        return experiment

    if main_function:
        return decorator(main_function)
    
    return decorator


def group_decorator(name=None):
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


class PystematicPlugin:

    def experiment_created(self, experiment):
        """Gives the plugin a chance to modify an experiment when it is created
        """
        pass

    def extend_api(self, api_object):
        """Gives the plugin a chance to modify the pystematic API.
        """
        pass

    def before_experiment(self, experiment, params):
        """Called before the main function of the experiment is executed.

        Args:
            experiment (Experiment): A handle to the experiment object.
            params (dict): Contains the values assigned to the parameters of the experiment.
        """
        pass

    def after_experiment(self):
        """Called after the experiment main function has returned. 
        """
        pass
