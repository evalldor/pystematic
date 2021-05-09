import typing
import sys

import pystematic
from . import parametric, core


def register_plugin(plugin_object, plugin_namespace):
    
    if hasattr(pystematic, plugin_namespace):
        raise ValueError(f"The namespace {plugin_namespace} is already in use")
    
    setattr(pystematic, plugin_namespace, plugin_object)

    sys.modules[f"pystematic.{plugin_namespace}"] = plugin_object


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

        param = core.Parameter(
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

        if isinstance(experiment, core.Experiment):
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
    api_object=None, 
    default_params=None,
    no_output_dir=False
):
    if callable(name):
        main_function = name
        name = None
    else:
        main_function = None

    def decorator(main_function):
        experiment = core.Experiment(
            api_object=api_object, 
            main_function=main_function, 
            default_params=default_params, 
            name=name, 
            defaults_override=defaults,
            no_output_dir=no_output_dir
        )

        existing_params = [param.name for param in experiment.get_parameters()]

        if inherit_params is not None:
            if not isinstance(inherit_params, (tuple, list)):
                experiments_to_inherit_from = [inherit_params]
            else:
                experiments_to_inherit_from = inherit_params

            for exp in experiments_to_inherit_from:
                if isinstance(exp, core.Experiment):
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


def group_decorator(name=None, experiment_decorator=None):
    if callable(name):
        main_function = name
        name = None
    else:
        main_function = None

    def decorator(main_function):
        group = core.ExperimentGroup(experiment_decorator, main_function, name=name)
        return group

    if main_function:
        return decorator(main_function)
    
    return decorator
