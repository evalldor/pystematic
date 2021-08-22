"""Parametric: next generation argument parsing and parameter handling

Handles all possible sources of parameter values:
- cli
- dict
- (file)
- env
- defaults

Clear cut, interceptable stages:
- cli parsing
- envvar extraction
- defaults insertion
- value validation


Motivation: 

A number of shortcommings in existing argparsing libs motivated the creation of
this lib. More specifically; handling many different sources of parameter
values, custom help text formatting, extensibility. 


Usage:

The main use of this lib is through (an instance of) the ParameterManager class.
It allows you to define parameters as well as read them from various sources.
"""


import collections
import typing
import typing_extensions
import sys
import re
import itertools
import os
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

        raise ValueError(f"Expected a value in {self.allowed_values}, got '{str(value)}'.")


class BooleanType:
    
    def __call__(self, value):
        if isinstance(value, bool):
            return value
        if str(value).lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif str(value).lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        elif value is None:
            return False
        
        raise ValueError(f"Boolean value expected, got {value}.")


class IdentityType:
    def __call__(self, value):
        return value


class DefaultParameterBehaviour:

    def after_init(self, param, **kwargs):
        pass

    def on_cli_value(self, param, flag:typing.Optional[str], value:typing.Any, result_dict:dict):
        self.on_value(param, value, result_dict)

    def on_value(self, param, value:typing.Any, result_dict:dict):
        result_dict[param.name] = value


class CompositeBehaviour:

    def __init__(self, *behaviours):
        self.behaviours = behaviours

    def after_init(self, param, **kwargs):
        for behavour in self.behaviours:
            behavour.after_init(param, **kwargs)

    def on_cli_value(self, param, flag:typing.Optional[str], value:typing.Any, result_dict:dict):
        for behavour in self.behaviours:
            behavour.on_cli_value(param, flag, value, result_dict)

    def on_value(self, param, value:typing.Any, result_dict:dict):
        for behavour in self.behaviours:
            behavour.on_value(param, value, result_dict)


class CliHelpBehaviour(DefaultParameterBehaviour):

    def __init__(self, param_manager):
        self.param_manager = param_manager

    def on_cli_value(self, param, flag:typing.Optional[str], value:typing.Any, result_dict:dict):
        self.param_manager.print_cli_help()
        sys.exit(0)


class BooleanFlagBehaviour(DefaultParameterBehaviour):

    def after_init(self, param, **kwargs):
        param.nargs = 0
        param.type = BooleanType()
        param.default = False if param.default is None else param.default

        flags = [flag for flag in param.flags if flag.startswith("--")]

        if len(flags) == 0:
            raise ValueError(f"Error during initialization: expected at least one long flag, got '{param.flags}'.")

        negative_flags = []
        for flag in flags:
            negative_flags.append(f"--no-{flag[2:]}")

        param.flags += negative_flags
    
    def on_cli_value(self, param, flag: typing.Optional[str], value: typing.Any, result_dict: dict):
        if flag.startswith("--no-"):
            result_dict[param.name] = False
        else:
            result_dict[param.name] = True


class Parameter:
    """Represents a parameter. 
    """
    def __init__(
        self,
        name: str,
        flags: typing.Union[None, typing.List[str], str] = None,
        type: typing.Callable[[str], typing.Any] = str,
        nargs: typing.Union[typing_extensions.Literal["?"], typing_extensions.Literal["*"], typing_extensions.Literal["+"], int, None] = None,
        default: typing.Union[typing.Any, typing.Callable[[], typing.Any], None] = None,
        required: bool = False,
        envvar: typing.Union[str, None, typing_extensions.Literal[False]] = None,

        help: typing.Optional[str] = None,
        default_help: typing.Optional[str] = None,
        hidden: bool = False,
        metavar: typing.Optional[str] = None,

        behaviour: DefaultParameterBehaviour = None,

        **extension_attributes
    ):

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
        if nargs not in (None, OPTIONAL, ZERO_OR_MORE, ONE_OR_MORE) and (not isinstance(nargs, int) or nargs < 0):
            raise ValueError(f"Invalid nargs value '{nargs}': must be a non negative integer or one of "
                             f"'{(None, OPTIONAL, ZERO_OR_MORE, ONE_OR_MORE)}'.")

        if envvar is not None:
            if not isinstance(envvar, str):
                raise ValueError(f"Invalid envvar type: expected a string.")

        if behaviour is None:
            behaviour = DefaultParameterBehaviour()

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

        self.behaviour = behaviour

        self.behaviour.after_init(self, **extension_attributes)

    def set_cli_value(self, flag:typing.Optional[str], value:typing.Any, result_dict:dict):
        self.behaviour.on_cli_value(self, flag, self.convert_value(value), result_dict)

    def set_value(self, value:typing.Any, result_dict:dict):
        self.behaviour.on_value(self, self.convert_value(value), result_dict)

    def convert_value(self, value):
        value = self._convert_to_correct_nargs(value)
        return self._convert_to_correct_type(value)

    def get_long_flags(self) -> typing.List[str]:
        return [flag for flag in self.flags if flag.startswith("--")]

    def get_short_flags(self) -> typing.List[str]:
        return [flag for flag in self.flags if re.match(r"^-[^-]", flag)]

    def get_envvar(self, prefix=None):
        if self.envvar is False:
            return None
        
        name = self.name.upper() if self.envvar is None else self.envvar
        full_name = name if prefix is None else prefix+name

        return full_name

    def _convert_to_correct_type(self, value):

        if isinstance(value, (list, tuple)):
            return list(map(lambda val: None if val is None else self.type(val), value))
        
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


class ParamValueDict(collections.UserDict):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.name_to_param_map = {param.name: param for param in params}

    def get_params(self):
        return self.params

    def set_value_by_name(self, name, value):
        if name in self.name_to_param_map:
            param = self.name_to_param_map[name]
            self.set_value(param, value)
        else:
            self[name] = value

    def set_value(self, param, value):
        try:
            param.set_value(value, self)
        except Exception as e:
            raise ValueError(f"Error when setting value for param '{param.name}': {e}") from None

    def set_cli_value(self, param, flag, value):
        try:
            param.set_cli_value(flag, value, self)
        except Exception as e:
            raise ValueError(f"Error when setting value for param '{param.name}': {e}") from None


@dataclasses.dataclass
class _ParamContainer:
    """Used to store some config together with a param
    """
    param: Parameter = None
    cli_enabled: bool = True
    cli_only: bool = False
    cli_positional: bool = False


class ParameterGroup:

    def __init__(self, name, param_manager):
        pass

    def add_parameter(self):
        pass

    def add_parameters(self):
        pass

    def add_param(self):
        pass


class ParameterManager:
    """The Parameter Manager essentialy consists of a set of parameters, and a
    set of operations to perform with those paramters, such as parsing them from
    cli, etc.
    """
    def __init__(
        self, 
        defaults_override={}, 
        env_prefix=None, 
        env_value_separators=":,",
        cli_help_formatter=None,
        add_cli_help_option=True,
    ) -> None:

        self._parameters: list[_ParamContainer] = []

        self.defaults_override = defaults_override

        self.env_prefix = env_prefix
        self.env_value_separators = env_value_separators

        self.cli_help_formatter = cli_help_formatter or CliHelpFormatter()

        if add_cli_help_option:
            self.add_param(
                name="__help__", 
                flags=["-h", "--help"],
                nargs=0,
                help="Show this help message and exit.",
                behaviour=CliHelpBehaviour(self),
                cli_only=True
            )

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
        flags: typing.Union[None, typing.List[str], str] = None,
        type: typing.Callable[[str], typing.Any] = str,
        nargs: typing.Union[typing_extensions.Literal["?"], typing_extensions.Literal["*"], typing_extensions.Literal["+"], int, None] = None,
        default: typing.Union[typing.Any, typing.Callable[[], typing.Any], None] = None,
        required: bool = False,
        envvar: typing.Optional[str] = None,

        help: typing.Optional[str] = None,
        default_help: typing.Optional[str] = None,
        hidden: bool = False,
        metavar: typing.Optional[str] = None,
        
        behaviour: DefaultParameterBehaviour = None,
        
        cli_only=False, 
        cli_positional=False, 
        cli_enabled=True,

        **kwargs
    ):
        param = Parameter(
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
            
            behaviour=behaviour,
            
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


    def validate_values(self, result_dict):
        for param in self.get_parameters():
            if param.name not in result_dict:
                raise ValueError(f"Missing value for param '{param.name}'.")
            
            if param.required and result_dict[param.name] is None:
                raise ValueError(f"Parameter '{param.name}' is required.")

    def from_cli(self, argv=None, defaults=True, env=True):
        result_dict = ParamValueDict(self.get_parameters())

        if argv is None:
            argv = sys.argv[1:]

        cli_results = parse_args(argv, self.get_cli_positionals(), self.get_cli_optionals())
        
        for param, flag, value in cli_results:
            result_dict.set_cli_value(param, flag, value)

        if env:
            self.add_env_values(result_dict)
        if defaults:
            self.add_defaults(result_dict)
        
        self.validate_values(result_dict)

        return dict(result_dict)

    def from_shared_cli(self, argv=None, defaults=True, env=True):
        result_dict = ParamValueDict(self.get_parameters())

        if argv is None:
            argv = sys.argv[1:]

        cli_results, argv_rest = parse_shared_args(argv, self.get_cli_positionals(), self.get_cli_optionals())
        
        for param, flag, value in cli_results:
            result_dict.set_cli_value(param, flag, value)

        if env:
            self.add_env_values(result_dict)
        if defaults:
            self.add_defaults(result_dict)

        self.validate_values(result_dict)

        return dict(result_dict), argv_rest

    def from_dict(self, values, defaults=True, env=True):
        result_dict = ParamValueDict(self.get_parameters())

        self.add_dict_values(values, result_dict)

        if env:
            self.add_env_values(result_dict)
        if defaults:
            self.add_defaults(result_dict)
        
        self.validate_values(result_dict)

        return dict(result_dict)

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
       

        if no_style:
            theme = Theme({}, inherit=False)
        
        else:
             theme = Theme({
                "metavar" : "dark_sea_green4",
                "heading": "bold",
                "flag": "cyan",
                "help_heading": "bold",
                "default_value": "white"

            }, inherit=False)

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

        if len(positionals) > 0:
            self._print_positionals_help(positionals)
            print()
        
        if len(optionals) > 0:
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
            padding=(0, 1, 0, 0),
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
            padding=(0, 1, 0, 0),
            show_header=False,
            show_footer=False,
            show_edge=False,
            width=100
        )

        grid.add_column()
        grid.add_column()
        grid.add_column()

        for param in optionals:
            if not param.hidden:
                flags = []
                for flag in param.get_short_flags() + param.get_long_flags():
                    flags.append(f"[flag]{flag}[/flag]")

                flags = ", ".join(flags)
                nargs = self._format_metavar(param)
                help = param.help or ""

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


class UnknownOptionsFlag(ParseError):
    pass


def parse_args(arg_list, positional_params, optional_params):

    # A note on terminology: The arg list consists of 'tokens'. A token may be
    # either a 'flag' or a 'value'. A 'flag' indicates an option, and a 'value'
    # is simply a value that may be assigned to a parameter. 

    # In the first step, the parser eats values from arg_list. All encountered
    # options are parsed right away, but any positional values are saved for the
    # next step.

    # The next step tries to assign all extracted positional values to their
    # corresponding parameters. There is an inherent ambiguity here when there
    # are several positional parameters that can consume an arbitrary number of
    # values. This ambiguity is resolved by letting each parameter in turn -
    # from left to right - consume as many values as they can, under the
    # constraint that the succeeding parameters can be assigned the minimal
    # number of values they require. The semantics are the same as in argparse.

    arg_list = list(arg_list) # Make mutable copy

    parsed_optionals = []

    max_num_positional_values = calculate_maximum_allowed_number_of_positional_values(positional_params)

    # Parse optionals
    parsed_optionals.extend(consume_optionals(optional_params, arg_list))

    # Parse one chunk of positional values
    positional_values, end_options_token_encountered = consume_positional_values(arg_list, max_num_positional_values, False)

    if not end_options_token_encountered:
        # Parse any remaining optionals
        parsed_optionals.extend(consume_optionals(optional_params, arg_list))
    
    # Any remaining values are unrecognized 
    if len(arg_list) > 0:
        parsing_error(arg_list, "Unrecognized value.")

    if len(positional_params) > 0:
        parsed_positionals = assign_values_to_positionals(positional_params, positional_values)
    else:
        parsed_positionals = []

    return parsed_positionals + parsed_optionals


def parse_known_args(arg_list, positional_params, optional_params):
    # This function stops parsing at the first unrecognized value, and returns
    # the remainder of the arg_list together with the parsed params. It stops
    # either when it encounters an unrecognized option or a positional value
    # that it cannot consume due to already having consumed the maximum number
    # of positional values.

    arg_list = list(arg_list) # Make mutable copy

    parsed_optionals = []

    max_num_positional_values = calculate_maximum_allowed_number_of_positional_values(positional_params)

    try:
        # Parse optionals
        parsed_optionals.extend(consume_optionals(optional_params, arg_list))

        # Parse one chunk of positional values
        positional_values, end_options_token_encountered = consume_positional_values(arg_list, max_num_positional_values, False)

        if not end_options_token_encountered:
            # Parse any remaining optionals
            parsed_optionals.extend(consume_optionals(optional_params, arg_list))
    except UnknownOptionsFlag:
        pass

    if len(positional_params) > 0:
        parsed_positionals = assign_values_to_positionals(positional_params, positional_values)
    else:
        parsed_positionals = []

    return parsed_positionals + parsed_optionals, arg_list


def parse_intermixed_args(arg_list, positional_params, optional_params):
    # This function allows intermixing positional args with options

    arg_list = list(arg_list) # Make mutable copy

    parsed_optionals = []
    positional_values = []
    max_num_positional_values = calculate_maximum_allowed_number_of_positional_values(positional_params)
    end_options_token_encountered = False

    while len(arg_list) > 0:
        if not end_options_token_encountered:
            parsed_optionals.extend(consume_optionals(optional_params, arg_list))

        if len(positional_values) >= max_num_positional_values:
            break

        values, end_options_token_encountered = consume_positional_values(
            arg_list, 
            max_num_values_to_consume=max_num_positional_values - len(positional_values), 
            end_options_token_encountered=end_options_token_encountered
        )
        positional_values.extend(values)
        

    # Any remaining values are unrecognized 
    if len(arg_list) > 0:
        parsing_error(arg_list, "Unrecognized value.")

    if len(positional_params) > 0:
        parsed_positionals = assign_values_to_positionals(positional_params, positional_values)
    else:
        parsed_positionals = []

    return parsed_positionals + parsed_optionals


def parse_known_intermixed_args(arg_list, positional_params, optional_params):
    # This function allows intermixing positional args with options

    arg_list = list(arg_list) # Make mutable copy

    parsed_optionals = []
    positional_values = []
    max_num_positional_values = calculate_maximum_allowed_number_of_positional_values(positional_params)
    end_options_token_encountered = False

    try:
        while len(arg_list) > 0:
            if not end_options_token_encountered:
                parsed_optionals.extend(consume_optionals(optional_params, arg_list))
            
            if len(positional_values) >= max_num_positional_values:
                break

            values, end_options_token_encountered = consume_positional_values(
                arg_list, 
                max_num_values_to_consume=max_num_positional_values - len(positional_values), 
                end_options_token_encountered=end_options_token_encountered
            )
            positional_values.extend(values)
    except UnknownOptionsFlag:
        pass

    if len(positional_params) > 0:
        parsed_positionals = assign_values_to_positionals(positional_params, positional_values)
    else:
        parsed_positionals = []

    return parsed_positionals + parsed_optionals, arg_list


def parse_shared_args(arg_list, positional_params, optional_params):
    # This function is intended to be used when there are subcommands present.
    # It parses a set of options and a chunk of positonal values, and leaves the
    # remainder for subcommands

    arg_list = list(arg_list) # Make mutable copy

    parsed_optionals = []

    max_num_positional_values = calculate_maximum_allowed_number_of_positional_values(positional_params)

    # Parse optionals
    parsed_optionals.extend(consume_optionals(optional_params, arg_list))

    # Parse one chunk of positional values
    positional_values, end_options_token_encountered = consume_positional_values(arg_list, max_num_positional_values, False)

    if len(positional_params) > 0:
        parsed_positionals = assign_values_to_positionals(positional_params, positional_values)
    else:
        parsed_positionals = []

    return parsed_positionals + parsed_optionals, arg_list


def consume_optionals(optional_params, arg_list):
    parsed_optionals = []

    while len(arg_list) > 0 and is_flag(arg_list[0]):
        parsed_optionals.extend(parse_optional(optional_params, arg_list))

    return parsed_optionals


def consume_positional_values(arg_list, max_num_values_to_consume, end_options_token_encountered):
    positional_values = []
    
    while (
        len(arg_list) > 0 and 
        len(positional_values) < max_num_values_to_consume and 
        (not is_flag(arg_list[0]) or end_options_token_encountered)
    ):
        if not end_options_token_encountered and arg_list[0] == "--":
            end_options_token_encountered = True
            arg_list.pop(0)
        else:
            positional_values.append(arg_list.pop(0))

    return positional_values, end_options_token_encountered


def parsing_error(arg_list, msg):
    # TODO: use arg list in error reporting
    raise ParseError(msg)


def calculate_maximum_allowed_number_of_positional_values(positional_params):
    max_num_values = 0

    for param in positional_params:
        if param.nargs == OPTIONAL:
            max_num_values += 1
        elif param.nargs == ZERO_OR_MORE:
            max_num_values += float("inf")
        elif param.nargs == ONE_OR_MORE:
            max_num_values += float("inf")
        else:
            max_num_values += param.nargs or 1
    
    return max_num_values


def is_flag(token):
    return token != "--" and re.match(r"^-\D", token)


def is_value(token):
    return token != "--" and not is_flag(token)


def parse_optional(optional_params, arg_list):

    def get_param_for_flag(flag):
        for param in optional_params:
            if param.flags is not None and flag in param.flags:
                return param
        
        raise UnknownOptionsFlag(f"Unrecognized option '{flag}'.")

    # Postpone popping the flag until we are sure we can consume it

    if "=" in arg_list[0]:
        flag, value = arg_list[0].split("=", 1)
        values = [value]
    else:
        flag = arg_list[0]
        values = []

    parsed_options = []

    # Flags can be either long or short (e.g. -l, --long)

    if re.match(r"^--", flag): # Handle long flags
        param = get_param_for_flag(flag)
        arg_list.pop(0)

        values += consume_optional_values(arg_list, param.nargs, num_already_consumed_values=len(values))

        parsed_options.append(ParseResult(param, flag, values))

    elif re.match(r"^-[^-]", flag): # Handle short flags
        # Shorthand flags may be joined together 
        # (e.g: 'ls -la' is equal to 'ls -l -a')
        
        if len(flag) == 2: # We have a single shorthand flag
            param = get_param_for_flag(flag)
            arg_list.pop(0)

            values += consume_optional_values(arg_list, param.nargs, num_already_consumed_values=len(values))

            parsed_options.append(ParseResult(param, flag, values))

        else: # We have muliple shorthand flags
            # None of the combined flags may consume any values. 
            # TODO: Maybe change in future...

            if len(values) > 0:
                parsing_error(arg_list, "Unexpected value encountered. Combined shorthand "
                                        "options can not be assigned values.")

            all_flags = [f"-{char}" for char in flag[1:]]

            for flag in all_flags:
                arg = get_param_for_flag(flag)
                parsed_options.append(ParseResult(arg, flag, []))

            arg_list.pop(0)

    else:
        parsing_error(arg_list, "Invalid flag encountered. This should never happen.")

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


def assign_values_to_positionals(positional_params, positional_values):
    parsed_positionals = []

    num_assigned_values_per_param = collections.defaultdict(lambda: 0)
    # Begin with assigning the minimum number of values required for each param,
    # from left to right.
    
    def get_num_assigned_values():
        return sum(num_assigned_values_per_param.values())

    curr_param_index = 0
    
    while get_num_assigned_values() < len(positional_values) and curr_param_index < len(positional_values):
        param = positional_params[curr_param_index]

        if param.nargs == OPTIONAL:
            num_assigned_values_per_param[param] = 0
            curr_param_index += 1
        elif param.nargs == ZERO_OR_MORE:
            num_assigned_values_per_param[param] = 0
            curr_param_index += 1
        elif param.nargs == ONE_OR_MORE:
            num_assigned_values_per_param[param] = 1
            curr_param_index += 1
        else:
            min_nargs = param.nargs or 1
            if num_assigned_values_per_param[param] < min_nargs:
                num_assigned_values_per_param[param] += 1
            else:
                curr_param_index += 1

    # If there are any values left, we assign them to the variable length
    # parameters from left to right

    while get_num_assigned_values() < len(positional_values) and curr_param_index < len(positional_params):
        param = positional_params[curr_param_index]

        if param.nargs == OPTIONAL:
            if num_assigned_values_per_param[param] < 1:
                num_assigned_values_per_param[param] = 1

            curr_param_index += 1  
        elif param.nargs == ZERO_OR_MORE:
            num_assigned_values_per_param[param] += 1
        elif param.nargs == ONE_OR_MORE:
            num_assigned_values_per_param[param] += 1
        else:
            # all params with a specific number of values have already been
            # filled in the previous step
            curr_param_index += 1
    
    assert get_num_assigned_values() == len(positional_values), "Internal error, too many positional values passed to 'assign_values_to_positionals'."

    value_indices = list(itertools.accumulate(num_assigned_values_per_param.values()))
    value_indices.insert(0, 0)

    for i, param in enumerate(positional_params):
        parsed_positionals.append(ParseResult(
            param, 
            None, 
            positional_values[value_indices[min(i, len(value_indices)-1)]:value_indices[min(i+1, len(value_indices)-1)]]
        ))

    return parsed_positionals
