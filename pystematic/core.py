import argparse
import dataclasses
import typing

from . import parametric

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

@dataclasses.dataclass
class Parameter:

    name: str
    type: typing.Callable[[str], typing.Any] = str
    
    default: typing.Union[typing.Any, typing.Callable[[], typing.Any], None] = None
    required: bool = False
    choices: list[typing.Any] = None
    is_flag: bool = False
    multiple: bool = False
    allow_from_file: bool = True
    envvar: typing.Union[str, None, typing.Literal[False]] = None

    help: typing.Optional[str] = None
    default_help: typing.Optional[str] = None


def _create_parameter(param):
    cls = parametric.Parameter

    if param.is_flag:
        if param.choices is not None:
            raise ValueError(f"Error in parameter declaration for '{param.name}': 'is_flag' is incompatible with 'choices'.")
        
        if param.multiple:
            raise ValueError(f"Error in parameter declaration for '{param.name}': 'is_flag' is incompatible with 'multiple'.")
        
        cls = parametric.BooleanFlagParameter
    else:

        type = param.type
        nargs = None
        
        if param.choices is not None:
            type = parametric.ChoiceType(param.choices)
        elif param.type == bool:
            type = parametric.BooleanType

        if param.multiple:
            nargs = "*"

        param = cls(
            name=param.name,
            type=type,

            required=param.required,
            default=param.default,
            nargs=nargs,

            help=param.help,
            default_help=param.default_help
        )

        param.allow_from_file = param.allow_from_file

        return param