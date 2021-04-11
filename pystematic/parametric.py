"""parametric: advanced parameter and args handling

clear cut stages:
- cli parsing
- envvar extraction
- defaults insertion
- value validation

param sources:
- cli
- dict
- env
- defaults
"""

__version__ = "0.1.0"
__all__ = []

import collections
import typing
import sys
import re
import itertools
import os
import contextlib
import dataclasses

OPTIONAL = "?"
ZERO_OR_MORE = "*"
ONE_OR_MORE = "+"


class ChoiceType:

    def __init__(self, allowed_values, case_sensitive=False):
        self.allowed_values = allowed_values
        self.case_sensitive = case_sensitive

    def __call__(self, value):
        
        for allowed_value in self.allowed_values:
            if self.case_sensitive:
                if str(allowed_value) == str(value):
                    return allowed_value
            else:
                if str(allowed_value).lower() == str(value).lower():
                    return allowed_value

        raise ValueError(f"Expected a value in '{self.allowed_values}', got '{str(value)}'.")


class BooleanType:
    
    def __call__(self, value):
        if isinstance(value, bool):
            return value
        if str(value).lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif str(value).lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        
        raise ValueError(f"Boolean value expected, got {value}.")


class IdentityType:
    def __call__(self, value):
        return value


class Parameter:

    def __init__(
        self,
        name: str,
        flags: typing.Union[None, list[str], str] = None,
        type: typing.Callable[[str], typing.Any] = str,
        nargs: typing.Union[typing.Literal["?"], typing.Literal["*"], typing.Literal["+"], int, None] = None,
        default: typing.Union[typing.Any, typing.Callable[[], typing.Any], None] = None,
        required: bool = False,
        envvar: typing.Union[str, None, typing.Literal[False]] = None,

        help: typing.Optional[str] = None,
        default_help: typing.Optional[str] = None,
        hidden: bool = False,
        metavar: typing.Optional[str] = None
    ) -> None:

        #
        # Validate name
        #
        if not isinstance(name, str):
            raise ValueError("Invalid name '{name}': expected a string.")

        if not name.isidentifier():
            raise ValueError("Invalid name '{name}': must be a valid python identifier.")
        
        #
        # Validate flags
        #
        if isinstance(flags, str):
            flags = [flags]
        elif flags is None or isinstance(flags, (list, tuple)):
            if flags is None or len(flags) == 0:
                flags = [f"--{name.lower().replace('_', '-')}"]

        else:
            raise ValueError(f"Unrecognized type for 'flags' ({flags}) should be list, str or None.")

        for flag in flags:
            if not isinstance(flag, str):
                raise ValueError(f"Invalid flag '{flag}': expected a string.")

            if len(flag) == 0:
                raise ValueError(f"Got an empty flag for param {name}.")

            if flag[0] != "-":
                raise ValueError(f"Invalid flag '{flag}': must start with the prefix '-'.")
            
            if re.match(r"^-[^-]{2,}", flag): 
                raise ValueError(f"Invalid flag '{flag}': short flags can only be a single character.")

            if re.match(r"^-\d", flag):
                raise ValueError(f"Invalid flag '{flag}': short flags cannot be numeric "
                                  "(it would result in ambiguity with negative numbers).")

        #
        # Validate type
        #
        if type is None:
            type = IdentityType()

        elif type is bool:
            type = BooleanType()

        if not callable(type):
            raise ValueError("Invalid type '{type}': must be a callable.")

        

        #
        # Validate nargs
        #
        if nargs not in (None, OPTIONAL, ZERO_OR_MORE, ONE_OR_MORE) and not isinstance(nargs, int):
            raise ValueError(f"Invalid nargs value '{nargs}': must be an int or one of "
                             f"'{(None, OPTIONAL, ZERO_OR_MORE, ONE_OR_MORE)}'.")

        if envvar is not None:
            if not isinstance(envvar, str):
                raise ValueError(f"Invalid envvar type: expected a string.")

        self.name = name
        self.flags = flags
        self.type = type
        self.nargs = nargs
        self.default = default
        self.required = required
        self.envvar = envvar

        self.help = help
        self.default_help = default_help
        self.hidden = hidden
        self.metavar = metavar

    def set_cli_value(self, flag:typing.Optional[str], value:typing.Any, result_dict:dict):
        self.set_value(value, result_dict)

    def set_value(self, value:typing.Any, result_dict:dict):
        result_dict[self.name] = self.convert_value(value)

    def convert_value(self, value):
        value = self._convert_to_correct_nargs(value)
        return self._convert_to_correct_type(value)

    def get_long_flags(self) -> list[str]:
        return [flag for flag in self.flags if flag.startswith("--")]

    def get_short_flags(self) -> list[str]:
        return [flag for flag in self.flags if re.match(r"^-[^-]", flag)]

    def get_envvar(self, prefix=None):
        if self.envvar is False:
            return None
        
        name = self.name.upper() if self.envvar is None else self.envvar
        full_name = name if prefix is None else prefix+name

        return full_name

    def _convert_to_correct_type(self, value):

        if isinstance(value, (list, tuple)):
            converted_values = []
        
            for val in value:
                converted_values.append(None if val is None else self.type(val))
            
            return converted_values
        
        return None if value is None else self.type(value)
            
    def _convert_to_correct_nargs(self, value):
        if isinstance(value, (list, tuple)):
            if self.nargs in (None, OPTIONAL, 0) and len(value) == 0:
                final_value = None
            elif self.nargs in (None, OPTIONAL, 0) and len(value) == 1:
                final_value = value[0]
            else:
                final_value = value
        else:
            final_value = value

        self._validate_nargs(final_value)

        return final_value

    def _validate_nargs(self, value):
        if self.nargs in (None, OPTIONAL, 0):
            if isinstance(value, (list, tuple)):
                raise ValueError(f"Expected a single value, got a collection of length {len(value)}.")
        elif value is not None:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"Expected a list of values, got '{value}'.")
            
            if self.nargs == ONE_OR_MORE:
                if len(value) == 0:
                    raise ValueError(f"Expected at least one value.")
            
            elif self.nargs != ZERO_OR_MORE:
                if len(value) != self.nargs:
                    raise ValueError(f"Expected a list of exactly {self.nargs} values, got {len(value)}.")

    def __str__(self) -> str:
        return f"Parameter(name={self.name}, flags={self.flags})"

    def __repr__(self) -> str:
        return str(self)


class CliHelpParameter(Parameter):

    def __init__(self, param_manager):
        super().__init__(
            name="__help__", 
            flags=["-h", "--help"],
            nargs=0,
            help="Show this help message and exit."
        )

        self.param_manager = param_manager

    def set_cli_value(self, flag:typing.Optional[str], value:typing.Any, result_dict:dict):
        self.param_manager.print_cli_help()
        sys.exit(0)


class BooleanFlagParameter(Parameter):

    def __init__(self, **kwargs):
        
        kwargs.update({
            "default": False if kwargs["default"] is None else kwargs["default"],
            "type": bool,
            "nargs": 0
        })

        super().__init__(**kwargs)

        flags = [flag for flag in self.flags if flag.startswith("--")]

        if len(flags) == 0:
            raise ValueError(f"Error during initialization: expected at least one long flag, got '{self.flags}'.")

        negative_flags = []
        for flag in flags:
            negative_flags.append(f"--no-{flag[2:]}")

        self.flags += negative_flags

    def set_cli_value(self, flag: typing.Optional[str], value: typing.Any, result_dict: dict):
        if flag.startswith("--no-"):
            result_dict[self.name] = False
        else:
            result_dict[self.name] = True

 
class ParamValues(collections.UserDict):

    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.name_to_param_map = {param.name: param for param in params}
        self.scopes = []

    @contextlib.contextmanager
    def scope(self, name):
        self.scopes.insert(0, name)
        yield
        self.scopes.pop(0)

    def set_value(self, param, value):
        with self.scope(param.name):
            param.set_value(value, self)

    def set_cli_value(self, param, flag, value):
        with self.scope(param.name):
            param.set_cli_value(flag, value, self)

    def __setitem__(self, name, value):
        if name not in self.name_to_param_map or (len(self.scopes) > 0 and self.scopes[0] == name):
            super().__setitem__(name, value)
        else:
            param = self.name_to_param_map[name]
            with self.scope(name):
                param.set_value(value, self)


@dataclasses.dataclass
class _ParamContainer:
    """Used to store some config together with a param
    """
    param:Parameter = None
    cli_enabled:bool = True
    cli_only:bool = False
    cli_positional:bool = False


class ParameterManager:

    def __init__(
        self, 
        defaults_override={}, 
        env_prefix=None, 
        env_value_separators=":,",
        cli_help_formatter=None,
        cli_allow_intermixed_args=False
    ) -> None:

        self._parameters: list[_ParamContainer] = []

        self.defaults_override = defaults_override

        self.env_prefix = env_prefix
        self.env_value_separators = env_value_separators

        self.cli_help_formatter = cli_help_formatter or CliHelpFormatter()
        self.cli_help_option = CliHelpParameter(self)
        self.cli_allow_intermixed_args = cli_allow_intermixed_args

    def add_parameter(self, param:Parameter, cli_only=False, cli_positional=False, cli_enabled=True):
        if cli_only and not cli_enabled:
            raise ValueError("'cli_only' and 'cli_enabled' may not be contradictive.")

        for existing_param in self.get_all_parameters():
            if existing_param.name == param.name:
                raise ValueError(f"Error when adding parameter '{param.name}': a parameter with the same name "
                                  "has already been added.")
            
            clashing_flags = set(existing_param.flags).intersection(param.flags)
            if len(clashing_flags) > 0:
                raise ValueError(f"Error when adding parameter '{param.name}': the parameter {existing_param.name} "
                                 f"also has flags '{clashing_flags}'.")

        self._parameters.append(_ParamContainer(
            param=param,
            cli_enabled=cli_enabled,
            cli_positional=cli_positional,
            cli_only=cli_only
        ))

    def add_param(
        self,
        name: str,
        flags: typing.Union[None, list[str], str] = None,
        type: typing.Callable[[str], typing.Any] = str,
        nargs: typing.Union[typing.Literal["?"], typing.Literal["*"], typing.Literal["+"], int, None] = None,
        default: typing.Union[typing.Any, typing.Callable[[], typing.Any], None] = None,
        required: bool = False,
        envvar: typing.Optional[str] = None,

        help: typing.Optional[str] = None,
        default_help: typing.Optional[str] = None,
        hidden: bool = False,
        metavar: typing.Optional[str] = None,
        
        cls: Parameter = Parameter,
        
        cli_only=False, 
        cli_positional=False, 
        cli_enabled=True,

        **kwargs
    ):
        param = cls(
            name=name,
            flags=flags,
            type=type,
            nargs=nargs,
            default=default,
            required=required,
            envvar=envvar,

            help=help,
            default_help=default_help,
            hidden=hidden,
            metavar=metavar,
            
            **kwargs
        )

        self.add_parameter(param, cli_only, cli_positional, cli_enabled)

        return param

    def get_all_parameters(self):
        return [c.param for c in self._parameters]

    def get_parameters(self):
        return [c.param for c in self._parameters if not c.cli_only]

    def get_cli_optionals(self):
        return [c.param for c in self._parameters if c.cli_enabled and not c.cli_positional]

    def get_cli_positionals(self):
        return [c.param for c in self._parameters if c.cli_enabled and c.cli_positional]

    def add_defaults(self, result_dict):
        
        for param in self.get_parameters():
            
            if param.name not in result_dict or result_dict[param.name] is None:
                if param.name in self.defaults_override:
                    default = self.defaults_override[param.name]
                else:
                    default = param.default
                
                default_value = default() if callable(default) else default
                result_dict.set_value(param, default_value)

    def add_env_values(self, result_dict):

        for param in self.get_parameters():
            
            if param.name not in result_dict or result_dict[param.name] is None:
                envvar = param.get_envvar(self.env_prefix)

                if envvar and envvar in os.environ:
                    values = re.split(
                        "|".join([re.escape(char) for char in self.env_value_separators]), 
                        os.environ[envvar]
                    )
                    values = [v for v in values if len(v) > 0]
                    
                    result_dict.set_value(param, values)

    def add_dict_values(self, values, result_dict):
        for param in self.get_parameters():
            if param.name in values:
                result_dict.set_value(param, values[param.name])

    def parse_cli_args(self, argv, result_dict):
        cli_results = parse_args(argv, self.get_cli_positionals(), self.get_cli_optionals(), self.cli_allow_intermixed_args)

        for param, flag, value in cli_results:
            result_dict.set_cli_value(param, flag, value)

    def validate_values(self, result_dict):
        for param in self.get_parameters():
            if param.name not in result_dict:
                raise ValueError(f"Missing value for param '{param.name}'.")
            
            if param.required and result_dict[param.name] is None:
                raise ValueError(f"Parameter '{param.name}' is required.")

    def from_cli(self, argv=None, defaults=True, env=True):
        result_dict = ParamValues(self.get_parameters())

        if argv is None:
            argv = sys.argv[1:]

        self.parse_cli_args(argv, result_dict)
        self.add_env_values(result_dict)
        self.add_defaults(result_dict)
        self.validate_values(result_dict)

        return result_dict

    def from_dict(self, values, defaults=True, env=True):
        result_dict = ParamValues(self.get_parameters())

        self.add_dict_values(values, result_dict)
        self.add_env_values(result_dict)
        self.add_defaults(result_dict)
        self.validate_values(result_dict)

        return result_dict

    def print_cli_usage(self):
        self.cli_help_formatter.print_usage(self.get_cli_positionals(), self.get_cli_optionals())

    def print_cli_help(self):
        self.cli_help_formatter.print_help(self.get_cli_positionals(), self.get_cli_optionals())


from rich.table import Table
from rich.padding import Padding
from rich.markup import escape
from rich.text import Text
from rich.console import Console
from rich.theme import Theme

class CliHelpFormatter:

    def __init__(self, no_style=False) -> None:
        theme = Theme({
            "metavar" : "dark_sea_green4",
            "heading": "bold",
            "flag": "cyan",
            "help_heading": "bold",
            "default_value": "white"

        }, inherit=False)

        if no_style:
            theme = Theme({}, inherit=False)

        self.console = Console(theme=theme)

    def print_usage(self, positionals, optionals):
        positionals_usage = []

        for param in positionals:
            positionals_usage.append(self._format_metavar(param, param.name))
    
        self.console.print("[heading]Usage:[/heading]")
        self.console.print(Padding(f"{os.path.basename(sys.argv[0])} [OPTIONS] {' '.join(positionals_usage)}", (0, 1)))

    def print_help(self, positionals, optionals):
        self.print_usage(positionals, optionals)
        print()
        self._print_positionals_help(positionals)
        print()
        self._print_optionals_help(optionals)

    def _format_metavar(self, param, metavar="<value>"):

        metavar = f"[metavar]{escape(metavar)}[/metavar]"

        if param.nargs == OPTIONAL:
            return f"[{metavar}]"

        elif param.nargs == ONE_OR_MORE:
            return f"{metavar} [{metavar} ...]"

        elif param.nargs == ZERO_OR_MORE:
            return f"[{metavar} ...]"

        else:
            if param.nargs is None:
                return f"{metavar}"
            else:
                return f"{' '.join([metavar]*param.nargs)}"      

    def _print_positionals_help(self, positionals):
        grid = Table(
            box=None,
            padding=(0, 1),
            show_header=False,
            show_footer=False,
            show_edge=False,
            width=100
        )

        grid.add_column()
        grid.add_column()

        for param in positionals:
            grid.add_row(param.name, param.help)

        self.console.print("[heading]Positional arguments:[/heading]")
        self.console.print(Padding(grid, (0, 1)))

    def _print_optionals_help(self, optionals):
        grid = Table(
            box=None,
            padding=(0, 1),
            show_header=False,
            show_footer=False,
            show_edge=False,
            width=100
        )

        grid.add_column()
        grid.add_column()
        grid.add_column()

        for param in optionals:
            flags = []
            for flag in param.get_short_flags() + param.get_long_flags():
                flags.append(f"[flag]{flag}[/flag]")

            flags = ", ".join(flags)
            nargs = self._format_metavar(param)
            help = param.help

            default_help = param.default_help or param.default
            
            if default_help is not None:
                help += f" [help_heading]Default:[/help_heading] [default_value]{default_help}[/default_value]."

            if param.required:
                help += " [help_heading]Required.[/help_heading]"

            grid.add_row(flags, nargs, help)

        self.console.print("[heading]Optional arguments:[/heading]")
        self.console.print(Padding(grid, (0, 1)))


ParseResult = collections.namedtuple("ParseResult", ["arg", "flag", "value"])

class ParseError(ValueError):
    pass


def parse_args(arg_list, positional_params, optional_params, allow_intermixed_args=False):

    # A note on terminology: The arg list consists of 'tokens'. A token may be
    # either a 'flag' or a 'value'. A 'flag' indicates an option, and a 'value'
    # is simply a value that may be assigned to a parameter. 

    # The first step simply extracts all optionals and positionals from the arg
    # list. The positionals are extracted in chunks, where the boundaries of the
    # chunks are determined by any encountered options. If
    # 'allow_intermixed_args' is False, there may only be a single positionals
    # chunk (i.e. the positional values may not be intermixed with options).
    # Otherwise, all chunks are joined together.

    arg_list = list(arg_list) # Make mutable copy

    parsed_optionals = []
    positional_values = []
    
    while len(arg_list) > 0:
        token = arg_list[0]

        if is_flag(token):
            parsed_optionals.extend(parse_optional(optional_params, arg_list))
        else:
            if not allow_intermixed_args and len(positional_values) > 0:
                # We have already extracted a chunk
                parsing_error(arg_list, "Unrecognized value.")
            
            positional_values.extend(consume_positional_values(arg_list))
   
    # The next step tries to assign all extracted positional values to their
    # corresponding parameters. There is an inherent ambiguity here when there
    # are several positional parameters that can consume an arbitrary number of
    # values. This ambiguity is resolved by letting the each parameter - from
    # left to right - consume as many values as they can, under the constraint
    # that the succeeding parameters can be assigned the minimal number of values
    # they require. The semantics are the same as in argparse.

    parsed_positionals = []

    if len(positional_params) > 0:
        num_assigned_values_per_param = {}
        max_num_values_per_param = {}
        for param in positional_params:
            if param.nargs == OPTIONAL:
                num_assigned_values_per_param[param] = 0
                max_num_values_per_param[param] = 1
            elif param.nargs == ZERO_OR_MORE:
                num_assigned_values_per_param[param] = 0
                max_num_values_per_param[param] = float("inf")
            elif param.nargs == ONE_OR_MORE:
                num_assigned_values_per_param[param] = 1
                max_num_values_per_param[param] = float("inf")
            else:
                num_assigned_values_per_param[param] = param.nargs or 1
                max_num_values_per_param[param] = param.nargs or 1

        if sum(num_assigned_values_per_param.values()) > len(positional_values):
            s = ""
            if sum(num_assigned_values_per_param.values()) < sum(max_num_values_per_param.values()):
                s = "at least " 

            parsing_error([], f"Too few positional args, expected {s}{sum(num_assigned_values_per_param.values())}, "
                               "got {len(positional_values)}.")
        
        curr_param_index = 0
            
        while sum(num_assigned_values_per_param.values()) < len(positional_values) and curr_param_index < len(positional_params):
            param = positional_params[curr_param_index]
            if num_assigned_values_per_param[param] < max_num_values_per_param[param]:
                num_assigned_values_per_param[param] += 1
            else:
                curr_param_index += 1
            
        if sum(num_assigned_values_per_param.values()) < len(positional_values):
            parsing_error([], f"Too many positional values, expected {sum(num_assigned_values_per_param.values())}, "
                               "got {len(positional_values)}.")

        
        value_indices = list(itertools.accumulate(num_assigned_values_per_param.values()))
        value_indices.insert(0, 0)

        for i, param in enumerate(positional_params):
            parsed_positionals.append(ParseResult(param, None, positional_values[value_indices[i]:value_indices[i+1]]))

    elif len(positional_values) > 0:
        parsing_error([], "Unrecognized arguments.")

    return parsed_positionals + parsed_optionals


def parsing_error(arg_list, msg):
    # TODO: use arg list in error reporting
    raise ParseError(msg)


def is_flag(token):
    return token != "--" and re.match(r"^-\D", token)


def is_value(token):
    return token != "--" and not is_flag(token)


def parse_optional(optional_params, arg_list):

    def get_param_for_flag(flag):
        for param in optional_params:
            if param.flags is not None and flag in param.flags:
                return param
        
        parsing_error(arg_list, "Unrecognized option.")

    # Postpone popping the flag until we are sure it will not be needed 
    # in reporting some possible error

    if "=" in arg_list[0]:
        flag, value = arg_list[0].split("=", 1)
        values = [value]
    else:
        flag = arg_list[0]
        values = []

    parsed_options = []

    # Flags can be either long or short (e.g. -l, --long)

    if re.match(r"^--", flag):
        param = get_param_for_flag(flag)
        arg_list.pop(0)

        values += consume_optional_values(arg_list, param.nargs, num_already_consumed_values=len(values))

        parsed_options.append(ParseResult(param, flag, values))

    elif re.match(r"^-[^-]", flag):
        # Shorthand flags may be joined together 
        # (e.g: 'ls -la' is equal to 'ls -l -a')
        
        if len(flag) == 2:
            # A single flag
            param = get_param_for_flag(flag)
            arg_list.pop(0)

            values += consume_optional_values(arg_list, param.nargs, num_already_consumed_values=len(values))

            parsed_options.append(ParseResult(param, flag, values))

        else:
            # None of the combined flags may consume any values. 
            # TODO: Maybe change in future...

            if len(values) > 0:
                parsing_error(arg_list, "Unexpected value encountered. Combined shorthand "
                                        "options can not be assigned values")

            all_flags = [f"-{char}" for char in flag[1:]]

            for flag in all_flags:
                arg = get_param_for_flag(flag)
                parsed_options.append(ParseResult(arg, flag, []))

            arg_list.pop(0)

    else:
        parsing_error(arg_list, "Invalid flag encountered. This should never happend.")

    return parsed_options


def consume_optional_values(arg_list, nargs, num_already_consumed_values=0):
    # This function consumes a number of values from the arg list determined by
    # 'nargs'. It always looks ahead to make sure that no optional flags are consumed.
    
    consumed_values = []

    if nargs == OPTIONAL:
        if len(arg_list) > 0 and is_value(arg_list[0]) and num_already_consumed_values == 0:
            consumed_values.append(arg_list.pop(0))
    elif nargs in [ZERO_OR_MORE, ONE_OR_MORE]:
        while len(arg_list) > 0 and is_value(arg_list[0]):
            consumed_values.append(arg_list.pop(0))
    else:
        if nargs is None:
            nargs = 1
        while len(arg_list) > 0 and is_value(arg_list[0]) and len(consumed_values) + num_already_consumed_values < nargs:
            consumed_values.append(arg_list.pop(0))
        
    return consumed_values


def consume_positional_values(arg_list):
    consumed_values = []

    while len(arg_list) > 0:
        if arg_list[0] == "--":
            # Everything after the first '--' are positional values.
            # The '--' token does not split positional values into chunks
            consumed_values += arg_list[1:]
            arg_list[:] = [] 
        elif is_value(arg_list[0]):
            consumed_values.append(arg_list.pop(0))
        else:
            break

    return consumed_values
