import argparse
import dataclasses
import typing

class Experiment:

    def __init__(self, main_function) -> None:
        self._main_function = main_function
        self._parameters = []
        self._defaults = {}
        pass

    def __call__(self, params):
        return self.run(params)

    def run(self, params):
        pass

    def cli(self):
        pass

T = typing.TypeVar("T")

@dataclasses.dataclass
class Parameter:

    name: str
    help: str = ""
    type: typing.Type[T] = str
    
    default: T = None
    required: bool = False
    
    choices: typing.List[typing.Any] = None
    is_flag: bool = False
    multiple: bool = False
    allow_from_file: bool = True
    envvar: str = None
    default_help: str = None



def _construct_argparser(parameters):
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@"
    )

    for param in parameters:
        _add_param_to_parser(param, parser)

    
    return parser

def _add_param_to_parser(param, parser):
    cli_name = f"--{param.name.replace('_', '-')}"

    if param.is_flag:
        if param.choices is not None:
            raise ValueError(f"Error in parameter declaration for '{param.name}': 'is_flag' is incompatible with 'choices'.")
        
        if param.multiple:
            raise ValueError(f"Error in parameter declaration for '{param.name}': 'is_flag' is incompatible with 'multiple'.")
        
        parser.add_argument(
            cli_name, 
            help=param.help,
            action=argparse.BooleanOptionalAction,
            default=param.default,
            dest=param.name
        )

    else:

        type = param.type
        nargs = None
        action = None
        
        if param.choices is not None:
            type = _choice_type(param.choices)
        elif param.type == bool:
            type = _bool_type

        if param.multiple:
            action = "extend"
            nargs = "*"

        parser.add_argument(
            cli_name, 
            help=param.help,
            type=type,
            default=param.default,
            choices=param.choices,
            action=action,
            nargs=nargs,
            dest=param.name
        )

def _bool_type(value):
    if isinstance(value, bool):
       return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _choice_type(choices):
    def nested(value):
        for choice in choices:
            if str(choice) == str(value):
                return choice
        
        raise argparse.ArgumentTypeError(f"Expected one of [{', '.join(choices)}]")

    return nested
