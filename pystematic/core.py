import sys
import functools
import multiprocessing
import importlib
import typing
import typing_extensions
import importlib_metadata
import logging

from . import parametric
from . import cli_help_formatters

logger = logging.getLogger("pystematic.core")

class PystematicApp:
    """A single instance of this class is created when pystematic initializes.
    Its main purpose is to centralize extension management. When an extension is
    initialized it is handed a reference to that instance.
    """

    def __init__(self) -> None:
        self._loaded_plugins = []

        self._experiment_created_callbacks = []
        self._before_experiment_callbacks = []
        self._after_experiment_callbacks = []
        
    def get_api_object(self):
        """Returns a handle to the ``pystematic`` global api. Extensions can
        retrieve the handle by calling this function, and modify the global api
        during initialization.

        Returns:
            module: The api available under the ``pystematic`` global namespace.
        """
        import pystematic
        return pystematic

    def load_all_plugins(self):
        """Finds and loads all plugins by searching entry points under the
        ``pystematic.plugins`` namespace. This function is internal and should
        only be called once.
        """

        plugin_entrypoints = importlib_metadata.entry_points().select(group='pystematic.plugins')
        
        for entry_point in plugin_entrypoints:
            logger.info(f"Loading plugin '{entry_point.name}'.")
            plugin = entry_point.load()
            self._loaded_plugins.append(plugin(self))

    def on_experiment_created(self, callback, priority=50):
        """Adds a callback to the 'experiment_created' event.

        Args:
            callback (function): A function that is called whenever a new experiment is defined. 
                It should take the created experiment as a single argument.
            priority (int, optional): The priority of the callback. Callbacks with lower priority 
                are called before those with higher. Defaults to 50.
        """
        self._experiment_created_callbacks.append((callback, priority))

    def on_before_experiment(self, callback, priority=50):
        """Adds a callback to the 'before_experiment' event. The event is
        triggered before the experiment main function is called.

        Args:
        
            callback (function): A function that is called before an experiment is
                run. It should take two arguments; the experiment, and a dict of
                parameter values.

            priority (int, optional): The priority of the callback. Callbacks with
                lower priority are called before those with higher. Defaults to 50.
        """
        self._before_experiment_callbacks.append((callback, priority))

    def on_after_experiment(self, callback, priority=50):
        """Adds a callback to the 'after_experiment' event. The event is
        triggered after the experiment main function has returned.

        Args:
            callback (function): A function that is called before an experiment is run. 
                The callback is not given any arguments.
            priority (int, optional): The priority of the callback. Callbacks with lower priority 
                are called before those with higher. Defaults to 50.
        """
        self._after_experiment_callbacks.append((callback, priority))


    def experiment_created(self, experiment):
        """Triggers the experiment_created event. Internal.
        """
        for callback, priority in sorted(self._experiment_created_callbacks, key=lambda x: x[1]):
            experiment = callback(experiment)
        
        return experiment

    def before_experiment(self, experiment, params):
        """Triggers the before_experiment event. Internal.
        """
        for callback, priority in sorted(self._before_experiment_callbacks, key=lambda x: x[1]):
            callback(experiment, params)

    def after_experiment(self):
        """Triggers the after_experiment event. Internal.
        """
        for callback, priority in sorted(self._after_experiment_callbacks, key=lambda x: x[1]):
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
    envvar: typing.Union[str, None, typing_extensions.Literal[False]] = None,

    help: typing.Optional[str] = None,
    default_help: typing.Optional[str] = None,
    hidden = False,
    behaviour = None,
):
    behaviours = [PystematicParameterBehaviour()]

    if behaviour is not None:
        behaviours.append(behaviour)

    
    if is_flag:
        if allowed_values is not None:
            raise ValueError(f"Error in parameter declaration for '{name}': 'is_flag' is incompatible with 'allowed_values'.")
        
        if multiple:
            raise ValueError(f"Error in parameter declaration for '{name}': 'is_flag' is incompatible with 'multiple'.")
        
        behaviours.append(parametric.BooleanFlagBehaviour())
    else:
        if allowed_values is not None:
            type = parametric.ChoiceType(allowed_values)
        elif type == bool:
            type = parametric.BooleanType()

    if multiple:
        nargs = "*"
    else:
        nargs = None
    
    return parametric.Parameter(
        name=name,
        type=type,

        required=required,
        default=default,
        nargs=nargs,
        envvar=envvar,

        help=help,
        default_help=default_help,
        hidden=hidden,
        behaviour=parametric.CompositeBehaviour(*behaviours),

        allow_from_file=allow_from_file
    )


class Experiment:
    """This is the class used to represent experiments. Note that you should not
    instantiate this class manually, but only through the
    :func:`pystematic.experiment` decorator.
    """ 

    def __init__(self, main_function, name=None, defaults_override={}):
        self.main_function = main_function
        self.name = name or main_function.__name__.lower().replace("_", "-")
        self._defaults_override = defaults_override
        
        self.param_manager = parametric.ParameterManager(
            defaults_override=defaults_override,
            add_cli_help_option=True
        )

    def add_parameter(self, param):
        """Adds a parameter to the experiment. Usually not used directly, but
        via the :func:`pystematic.parameter` decorator.

        Args:
            param (Parameter): The parameter to add.
        """
        self.param_manager.add_parameter(param)

    def get_parameters(self):
        """Returns a list of all parameters registered with this experiment.
        """
        return self.param_manager.get_parameters()

    def __call__(self, params):
        return self.run(params)

    def run(self, params):
        """Runs the experiment in the current process with the provided parameters.

        Args:
            params (dict): A dict containing the values for the parameters.
        """
        param_values = self.param_manager.from_dict(params)
        self._run_experiment(param_values)

    def cli(self, argv=None):
        """Runs the experiment by parsing the parameters from the command line.

        Args:
            argv (List[str], optional): A list of command line arguments. If None, will use 
                ``sys.argv``. Defaults to None.
        """
        if argv is None:
            argv = sys.argv[1:]

        param_values = self.param_manager.from_cli(argv)
        self._run_experiment(param_values)

    def run_in_new_process(self, params):
        """Runs the experiment in a new process with the parameters provided.
        Returns a handle to the process object used to run the experiment. If
        you want to wait for the experiment to finish you have to manually wait
        for the process to exit.

        Args:
            params (dict): A dict containing the values for the parameters.

        Returns:
            multiprocessing.Process: The process object used to run the experiment
        """
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
    """Use when you have many experiments and want to group them in some way.
    """
    def __init__(self, main_function, name=None):
        
        self.main_function = main_function
        self.name = name or main_function.__name__.lower().replace("_", "-")

        self.experiments = []

        self.param_manager = parametric.ParameterManager(
            add_cli_help_option=True,
            cli_help_formatter=cli_help_formatters.GroupHelpFormatter(self)
        )

        self.param_manager.add_param(
            name="experiment",
            help="The name of the experiment to run.",
            required=True,
            cli_positional=True
        )

    def experiment(self, name=None, inherit_params=None, defaults={}):
        """Creates an experiment and adds it to this group. See also :func:`pystematic.experiment`.
        """
        if callable(name):
            main_function = name
            name = None
        else:
            main_function = None

        if main_function:
            return _experiment_constructor(
                main_function, 
                name=name, 
                inherit_params=inherit_params, 
                defaults=defaults, 
                group=self
            )
        
        return functools.partial(
            _experiment_constructor, 
            name=name, 
            inherit_params=inherit_params, 
            defaults=defaults, 
            group=self
        )

    def group(self, name=None):
        """Creates a nested group. See also :func:`pystematic.group`
        """
        if callable(name):
            main_function = name
            name = None
        else:
            main_function = None

        if main_function:
            return _group_constructor(main_function, name=name, group=self)
        
        return functools.partial(_group_constructor, name=name, group=self)

    def add_experiment(self, experiment):
        """Adds an experiment to the group. Typically not called manually.

        Args:
            experiment (Experiment): The experiment to add
        """
        self.experiments.append(experiment)

    def cli(self, argv=None):
        """Runs the group by parsing the parameters from the command line. The
        first argument should be the name of the experiment to run.

        Args:
            argv (List[str], optional): A list of command line arguments. If None, will use 
                ``sys.argv``. Defaults to None.
        """
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
    envvar: typing.Union[str, None, typing_extensions.Literal[False]] = None,

    help: typing.Optional[str] = None,
    default_help: typing.Optional[str] = None,
    hidden = False,
    behaviour = None,
):
    """Adds a parameter to an experiment.

    Args:

        name (str): The name of the parameter. The name must be a valid python
            identifier.

        type (typing.Callable[[str], typing.Any], optional): The type of the
            parameter. Must be a callable that takes a single string value argument
            and returns the converted value. Defaults to str.

        default (typing.Union[typing.Any, typing.Callable[[], typing.Any], None],
            optional): The default value of the parameter. Can be either a value or
            a zero arguments callable that returns the value. Defaults to None.

        required (bool, optional): Set to True if this parameter is required. If a
            default value is set, the parameter will effectively no longer be
            'required', even if this options is set to True. Defaults to False.

        allowed_values (list[typing.Any], optional): If given, the value must be in
            the list of allowed values. Defaults to None.

        is_flag (bool, optional): When set to True, this parameter is assumed to be
            a boolean flag. A flag parameter does not need to be given a value on
            the command line. Its mere presence on the command line will
            automatically assign it the value True. Defaults to False.

        multiple (bool, optional): When set to True, the parameter is assumed to be
            a list of zero or more values. Defaults to False.

        allow_from_file (bool, optional): Controls whether it should be allowed to
            load a value for this parameter from a params file. Defaults to True.

        envvar (typing.Union[str, None, typing.Literal[False]], optional): Name of
            the environment variable that the value for this parameter may be read
            from. Defaults to None.

        help (typing.Optional[str], optional): A help text for the parameter that
            will be shown on the command line. Defaults to None.

        default_help (typing.Optional[str], optional): A help text for the default
            value. If None, the default help text will be created by calling
            ``str(default_value)``. Useful when the default value is a callable. 
            Defaults to None.
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
):
    """Creates a new experiment with the decorated function as the main function.

    Args:
        name (str, optional): Name of the experiment. Defaults to the name of the main function.
        inherit_params (Experiment, List[Experiment], optional): An experiment, or a list of experiments 
            to inherit parameters from. Defaults to None.
        defaults (dict, optional): A dict containing default values for parameters, will override any 
            default set in the parameter declaration. Defaults to {}.
    """
    if callable(name):
        main_function = name
        name = None
    else:
        main_function = None

    if main_function:
        return _experiment_constructor(
            main_function, 
            name=name, 
            inherit_params=inherit_params, 
            defaults=defaults
        )
    
    return functools.partial(
        _experiment_constructor, 
        name=name, 
        inherit_params=inherit_params, 
        defaults=defaults
    )


def group_decorator(name=None):
    """Used to group experiments. This decorator is used on a function. Note
    that the decorated function will never be called.

    Args:
        name (str, optional): The name of the group. Defaults to None.
    """
    if callable(name):
        main_function = name
        name = None
    else:
        main_function = None

    if main_function:
        return _group_constructor(main_function, name=name)
    
    return functools.partial(_group_constructor, name=name)


def _experiment_constructor(main_function, name=None, inherit_params=None, defaults={}, group=None):

    experiment = Experiment(
        main_function=main_function, 
        name=name, 
        defaults_override=defaults
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


def _group_constructor(main_function, name=None, group=None):
    new_group = ExperimentGroup(main_function, name=name)

    if group is not None: # nested groups
        group.add_experiment(new_group)
    
    return new_group
