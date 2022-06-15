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

A number of shortcomings in existing argparsing libs motivated the creation of
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


class Error(Exception):
    pass


class ValidationError(Error):
    pass


class ParamInitError(Error):
    pass

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

        raise ValidationError(f"Expected a value in {self.allowed_values}, got '{str(value)}'.")


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
        
        raise ValidationError(f"Boolean value expected, got {value}.")


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
            raise Error(f"Error during initialization: expected at least one long flag, got '{param.flags}'.")

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

        cli_only: bool = False, 
        cli_positional: bool = False, 
        cli_enabled: bool = True,

        **extension_attributes
    ):

        #
        # Validate name
        #
        if not isinstance(name, str):
            raise ParamInitError("Invalid name '{name}': expected a string.")

        if not name.isidentifier():
            raise ParamInitError("Invalid name '{name}': must be a valid python identifier.")
        
        #
        # Validate flags
        #
        if isinstance(flags, str):
            flags = [flags]
        elif flags is None or isinstance(flags, (list, tuple)):
            if flags is None or len(flags) == 0:
                flags = [f"--{name.lower().replace('_', '-')}"]

        else:
            raise ParamInitError(f"Unrecognized type for 'flags' ({flags}) should be list, str or None.")

        for flag in flags:
            if not isinstance(flag, str):
                raise ParamInitError(f"Invalid flag '{flag}': expected a string.")

            if len(flag) == 0:
                raise ParamInitError(f"Got an empty flag for param {name}.")

            if flag[0] != "-":
                raise ParamInitError(f"Invalid flag '{flag}': must start with the prefix '-'.")
            
            if re.match(r"^-[^-]{2,}", flag): 
                raise ParamInitError(f"Invalid flag '{flag}': short flags can only be a single character.")

            if re.match(r"^-\d", flag):
                raise ParamInitError(f"Invalid flag '{flag}': short flags cannot be numeric "
                                  "(it would result in ambiguity with negative numbers).")

        #
        # Validate type
        #
        if type is None:
            type = IdentityType()

        elif type is bool:
            type = BooleanType()

        if not callable(type):
            raise ParamInitError("Invalid type '{type}': must be a callable.")

        

        #
        # Validate nargs
        #
        if nargs not in (None, OPTIONAL, ZERO_OR_MORE, ONE_OR_MORE) and (not isinstance(nargs, int) or nargs < 0):
            raise ParamInitError(f"Invalid nargs value '{nargs}': must be a non negative integer or one of "
                             f"'{(None, OPTIONAL, ZERO_OR_MORE, ONE_OR_MORE)}'.")

        if envvar is not None:
            if not isinstance(envvar, str):
                raise ParamInitError(f"Invalid envvar type: expected a string.")

        if behaviour is None:
            behaviour = DefaultParameterBehaviour()

        if cli_only and not cli_enabled:
            raise ParamInitError("'cli_only' and 'cli_enabled' may not be contradictive.")

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

        self.cli_only = cli_only
        self.cli_positional = cli_positional
        self.cli_enabled = cli_enabled

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
                raise ValidationError(f"Expected a single value, got a collection of length {len(value)}.")
        elif value is not None:
            if not isinstance(value, (list, tuple)):
                raise ValidationError(f"Expected a list of values, got '{value}'.")
            
            if self.nargs == ONE_OR_MORE:
                if len(value) == 0:
                    raise ValidationError(f"Expected at least one value.")
            
            elif self.nargs != ZERO_OR_MORE:
                if len(value) != self.nargs:
                    raise ValidationError(f"Expected a list of exactly {self.nargs} values, got {len(value)}.")

    def __str__(self) -> str:
        return f"Parameter(name={self.name}, flags={self.flags})"

    def __repr__(self) -> str:
        return str(self)


class _ParamValueDict(collections.UserDict):
    # todo: go back to scopes and __set_item__
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
            raise Error(f"Error when setting value for param '{param.name}': {e}") from e

    def set_cli_value(self, param, flag, value):
        try:
            param.set_cli_value(flag, value, self)
        except Exception as e:
            raise Error(f"Error when setting value for param '{param.name}': {e}") from e


class OptionGroup:
    """Container for a set of related parameters. A parameter can belong to more
    than one group.
    """

    def __init__(self, name, help=None, parameters: typing.List[Parameter] = []):
        self.name = name
        self.help = help
        self._parameters: typing.List[Parameter] = []

        for param in parameters:
            self.add_parameter(param)

    def add_parameter(self, param: Parameter):
        if param.cli_positional:
            raise Error(f"Error when adding parameter '{param.name}' to group '{self.name}': "
                        "option groups cannot contain CLI positional parameters.")

        _check_for_clash_with_existing_params(param, self._parameters)

        self._parameters.append(param)

    def get_parameters(self):
        return [param for param in self._parameters if not param.cli_only]

    def get_cli_optionals(self):
        return [param for param in self._parameters if param.cli_enabled and not param.cli_positional]

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
        
        cli_only: bool = False, 
        cli_enabled: bool = True,

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
            
            cli_only=cli_only,
            cli_positional=False,
            cli_enabled=cli_enabled,

            **kwargs
        )

        self.add_parameter(param)

        return param


def _check_for_clash_with_existing_params(param, existing_params):

    for existing_param in existing_params:
        if existing_param.name == param.name:
            raise Error(f"Error when adding parameter '{param.name}': a parameter with the same name "
                        "has already been added.")
        
        clashing_flags = set(existing_param.flags).intersection(param.flags)
        if len(clashing_flags) > 0:
            raise Error(f"Error when adding parameter '{param.name}': the parameter {existing_param.name} "
                        f"also has flags '{clashing_flags}'.")


def _unique_params(*parameter_lists):
    unique_list = []

    for param_list in parameter_lists:
        for param in param_list:
            if param not in unique_list:
                unique_list.append(param)

    return unique_list


class ParameterManager:
    """The Parameter Manager essentialy consists of a set of parameters, and a
    set of operations to perform with those paramters, such as parsing them from
    cli.
    """
    def __init__(
        self, 
        defaults_override={}, 
        env_prefix: str = None, 
        env_value_separators: str = ":,",
        help_formatter = None,
        add_cli_help_option: bool = True,
    ) -> None:

        self._parameters: typing.List[Parameter] = []
        self._groups: typing.List[OptionGroup] = []

        self.defaults_override = defaults_override

        self.env_prefix = env_prefix
        self.env_value_separators = env_value_separators

        self.help_formatter = help_formatter or HelpFormatter(self)

        if add_cli_help_option:
            self.add_param(
                name="__help__", 
                flags=["-h", "--help"],
                nargs=0,
                help="Show this help message and exit.",
                behaviour=CliHelpBehaviour(self),
                cli_only=True
            )


    def add_parameter(self, param: Parameter):
        _check_for_clash_with_existing_params(param, self.get_all_parameters())

        self._parameters.append(param)

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
        
        cli_only: bool = False, 
        cli_positional: bool = False, 
        cli_enabled: bool = True,

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
            
            cli_only=cli_only,
            cli_positional=cli_positional,
            cli_enabled=cli_enabled,

            **kwargs
        )

        self.add_parameter(param)

        return param

    def add_group(self, group):
        # Copies the group and adds it. Subsequent changes to the group have no
        # effect on the already added group

        for existing_group in self._groups:
            if existing_group.name == group.name:
                raise Error(f"Error when adding group '{group.name}': a group with that "
                            "name has already been added.")

        group_copy = OptionGroup(group.name, group.help, group._parameters)

        for param in group.get_parameters():
            _check_for_clash_with_existing_params(param, self.get_all_parameters())

        self._groups.append(group_copy)

    def get_groups(self):
        return list(self._groups)

    def get_all_parameters_in_groups(self):
        return _unique_params(*[group._parameters for group in self._groups])

    def get_all_parameters(self):
        return _unique_params(self._parameters, *[group._parameters for group in self._groups])

    def get_all_ungrouped_parameters(self):
        return list(self._parameters)

    def get_ungrouped_cli_optionals(self):
        return [param for param in self.get_all_ungrouped_parameters() if param.cli_enabled and not param.cli_positional]

    def get_parameters(self):
        return [param for param in self.get_all_parameters() if not param.cli_only]

    def get_cli_only_parameters(self):
        return [param for param in self.get_all_parameters() if param.cli_only]

    def get_cli_optionals(self):
        return [param for param in self.get_all_parameters() if param.cli_enabled and not param.cli_positional]

    def get_cli_positionals(self):
        return [param for param in self.get_all_parameters() if param.cli_enabled and param.cli_positional]


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
                raise ValidationError(f"Missing value for param '{param.name}'.")

            if param.required and result_dict[param.name] is None:
                raise ValidationError(f"Parameter '{param.name}' is required.")

    def fail_on_unknown_values(self, result_dict):
        all_param_names = [param.name for param in self.get_all_parameters()]
        
        for name in result_dict.keys():
            
            if name not in all_param_names:
                raise ValidationError(f"'{name}' is not a recognized parameter name.")

    def from_cli(self, argv=None, defaults=True, env=True):
        result_dict = _ParamValueDict(self.get_parameters())

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
        result_dict = _ParamValueDict(self.get_parameters())

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

    def from_dict(self, values, defaults=True, env=True, fail_on_unknown_values=False):
        if(fail_on_unknown_values):
            self.fail_on_unknown_values(values)

        result_dict = _ParamValueDict(self.get_parameters())

        self.add_dict_values(values, result_dict)

        if env:
            self.add_env_values(result_dict)

        if defaults:
            self.add_defaults(result_dict)
        
        

        self.validate_values(result_dict)

        return dict(result_dict)

    def print_cli_usage(self):
        self.help_formatter.print_usage(self)

    def print_cli_help(self):
        self.help_formatter.print_help(self)

    def print_error(self, e: Exception):
        self.help_formatter.print_error(e)



from rich.table import Table
from rich.padding import Padding
from rich.markup import escape
from rich.text import Text
from rich.console import Console
from rich.theme import Theme


class HelpFormatter:

    def __init__(self, no_style=False) -> None:
        
        if no_style:
            theme = Theme({}, inherit=False)
        
        else:
             theme = Theme({
                "metavar" : "dark_sea_green4",
                "heading": "bold",
                "flag": "cyan",
                "help_heading": "bold",
                "default_value": "bold",
                "error": "bold red"

            }, inherit=False)

        self.console = Console(theme=theme)

    def print_usage(self, param_manager: ParameterManager):
        positionals = param_manager.get_cli_positionals()
        positionals_usage = []

        for param in positionals:
            positionals_usage.append(self._format_metavar(param, param.name))
    
        self.console.print("[heading]Usage:[/heading]")
        self.console.print(Padding(f"{os.path.basename(sys.argv[0])} [OPTIONS] {' '.join(positionals_usage)}", (0, 1)))

    def print_help(self, param_manager: ParameterManager):
        self.print_usage(param_manager)
        self.console.print()

        positionals = param_manager.get_cli_positionals()

        if len(positionals) > 0:
            self._print_positionals_help(positionals)
            self.console.print()

        optionals = param_manager.get_cli_optionals()
        
        if len(optionals) > 0:
            self._print_optionals_help(optionals)

    def print_error(self, error:Exception):
        self.console.print(f"[error]Error:[/error] {error}")

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
            padding=(0, 1, 1, 0),
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

                grid.add_row(flags, Padding(nargs, (0, 2)), help)

        self.console.print("[heading]Optional arguments:[/heading]")
        self.console.print(Padding(grid, (0, 1)))


_ParseResult = collections.namedtuple("_ParseResult", ["arg", "flag", "value"])


class _ArgList:
    """This is a convenience wrapper for the argument list used when parsing.
    The main purpose is to allow for detailed paring errors by having access to
    the whole argument list, and the currently processed token in the list.
    """

    def __init__(self, arg_list) -> None:
        self.arg_list = arg_list
        self.curr_pos = 0

    def has_remaining(self):
        """Returns num_remaining() > 0
        """
        return self.curr_pos < len(self.arg_list)

    def num_remaining(self):
        """Returns the numner of items not yet consumed.
        """
        return len(self.arg_list) - self.curr_pos

    def get_remaining(self):
        return self.arg_list[self.curr_pos:]

    def peek(self):
        """Returns the current item.
        """
        return self.arg_list[self.curr_pos]

    def consume(self):
        """Returns the current item and increments the counter by one.
        """
        item = self.arg_list[self.curr_pos]
        self.curr_pos += 1

        return item


class ParseError(Error):
    
    def __init__(self, message, arg_list) -> None:
        super().__init__(message)
        self.arg_list = arg_list


class UnknownOptionsFlag(ParseError):
    pass


def parse_args(arg_list, positional_params, optional_params):
    """Allows optionals to appear both before and after the chunk of positionals
    """

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

    arg_list = _ArgList(arg_list)

    parsed_optionals = []

    max_num_positional_values = _calculate_maximum_allowed_number_of_positional_values(positional_params)

    # Parse optionals
    parsed_optionals.extend(_consume_optionals(optional_params, arg_list))

    # Parse one chunk of positional values
    positional_values, end_options_token_encountered = _consume_positional_values(arg_list, max_num_positional_values, False)

    if not end_options_token_encountered:
        # Parse any remaining optionals
        parsed_optionals.extend(_consume_optionals(optional_params, arg_list))
    
    # Any remaining values are unrecognized 
    if arg_list.has_remaining():
        _parsing_error(arg_list, "Unrecognized value.")

    if len(positional_params) > 0:
        parsed_positionals = _assign_values_to_positionals(positional_params, positional_values)
    else:
        parsed_positionals = []

    return parsed_positionals + parsed_optionals


def parse_known_args(arg_list, positional_params, optional_params):
    # This function stops parsing at the first unrecognized value, and returns
    # the remainder of the arg_list together with the parsed params. It stops
    # either when it encounters an unrecognized option or a positional value
    # that it cannot consume due to already having consumed the maximum number
    # of positional values.

    arg_list = _ArgList(arg_list) # Make mutable copy

    parsed_optionals = []

    max_num_positional_values = _calculate_maximum_allowed_number_of_positional_values(positional_params)

    try:
        # Parse optionals
        parsed_optionals.extend(_consume_optionals(optional_params, arg_list))

        # Parse one chunk of positional values
        positional_values, end_options_token_encountered = _consume_positional_values(arg_list, max_num_positional_values, False)

        if not end_options_token_encountered:
            # Parse any remaining optionals
            parsed_optionals.extend(_consume_optionals(optional_params, arg_list))
    except UnknownOptionsFlag:
        pass

    if len(positional_params) > 0:
        parsed_positionals = _assign_values_to_positionals(positional_params, positional_values)
    else:
        parsed_positionals = []

    return parsed_positionals + parsed_optionals, arg_list.get_remaining()


def parse_intermixed_args(arg_list, positional_params, optional_params):
    # This function allows intermixing positional args with options

    arg_list = _ArgList(arg_list) # Make mutable copy

    parsed_optionals = []
    positional_values = []
    max_num_positional_values = _calculate_maximum_allowed_number_of_positional_values(positional_params)
    end_options_token_encountered = False

    while arg_list.has_remaining():
        if not end_options_token_encountered:
            parsed_optionals.extend(_consume_optionals(optional_params, arg_list))

        if len(positional_values) >= max_num_positional_values:
            break

        values, end_options_token_encountered = _consume_positional_values(
            arg_list, 
            max_num_values_to_consume=max_num_positional_values - len(positional_values), 
            end_options_token_encountered=end_options_token_encountered
        )
        positional_values.extend(values)
        

    # Any remaining values are unrecognized 
    if arg_list.has_remaining():
        _parsing_error(arg_list, "Unrecognized value.")

    if len(positional_params) > 0:
        parsed_positionals = _assign_values_to_positionals(positional_params, positional_values)
    else:
        parsed_positionals = []

    return parsed_positionals + parsed_optionals


def parse_known_intermixed_args(arg_list, positional_params, optional_params):
    # This function allows intermixing positional args with options

    arg_list = _ArgList(arg_list) # Make mutable copy

    parsed_optionals = []
    positional_values = []
    max_num_positional_values = _calculate_maximum_allowed_number_of_positional_values(positional_params)
    end_options_token_encountered = False

    try:
        while arg_list.has_remaining():
            if not end_options_token_encountered:
                parsed_optionals.extend(_consume_optionals(optional_params, arg_list))
            
            if len(positional_values) >= max_num_positional_values:
                break

            values, end_options_token_encountered = _consume_positional_values(
                arg_list, 
                max_num_values_to_consume=max_num_positional_values - len(positional_values), 
                end_options_token_encountered=end_options_token_encountered
            )
            positional_values.extend(values)
    except UnknownOptionsFlag:
        pass

    if len(positional_params) > 0:
        parsed_positionals = _assign_values_to_positionals(positional_params, positional_values)
    else:
        parsed_positionals = []

    return parsed_positionals + parsed_optionals, arg_list.get_remaining()


def parse_shared_args(arg_list, positional_params, optional_params):
    # This function is intended to be used when there are subcommands present.
    # It parses a set of options and a chunk of positonal values, and leaves the
    # remainder for subcommands

    arg_list = _ArgList(arg_list) # Make mutable copy

    parsed_optionals = []

    max_num_positional_values = _calculate_maximum_allowed_number_of_positional_values(positional_params)

    # Parse optionals
    parsed_optionals.extend(_consume_optionals(optional_params, arg_list))

    # Parse one chunk of positional values
    positional_values, end_options_token_encountered = _consume_positional_values(arg_list, max_num_positional_values, False)

    if len(positional_params) > 0:
        parsed_positionals = _assign_values_to_positionals(positional_params, positional_values)
    else:
        parsed_positionals = []

    return parsed_positionals + parsed_optionals, arg_list.get_remaining()


def _consume_optionals(optional_params, arg_list):
    parsed_optionals = []

    while arg_list.has_remaining() and _is_flag(arg_list.peek()):
        parsed_optionals.extend(_parse_optional(optional_params, arg_list))

    return parsed_optionals


def _consume_positional_values(arg_list, max_num_values_to_consume, end_options_token_encountered):
    positional_values = []
    
    while (
        arg_list.has_remaining() and 
        len(positional_values) < max_num_values_to_consume and 
        (not _is_flag(arg_list.peek()) or end_options_token_encountered)
    ):
        if not end_options_token_encountered and arg_list.peek() == "--":
            end_options_token_encountered = True
            arg_list.consume()
        else:
            positional_values.append(arg_list.consume())

    return positional_values, end_options_token_encountered


def _parsing_error(arg_list, msg):
    # TODO: use arg list in error reporting
    raise ParseError(msg, arg_list)


def _calculate_maximum_allowed_number_of_positional_values(positional_params):
    max_num_values = 0

    for param in positional_params:
        if param.nargs == OPTIONAL:
            max_num_values += 1
        elif param.nargs == ZERO_OR_MORE:
            max_num_values += float("inf")
        elif param.nargs == ONE_OR_MORE:
            max_num_values += float("inf")
        else:
            max_num_values += param.nargs if param.nargs is not None else 1
    
    return max_num_values


def _is_flag(token):
    return token != "--" and re.match(r"^-\D", token)


def _is_value(token):
    return token != "--" and not _is_flag(token)


def _parse_optional(optional_params, arg_list):

    def get_param_for_flag(flag):
        for param in optional_params:
            if param.flags is not None and flag in param.flags:
                return param
        
        raise UnknownOptionsFlag(f"Unrecognized option '{flag}'.", arg_list)

    # Postpone popping the flag until we are sure we can consume it

    if "=" in arg_list.peek():
        flag, value = arg_list.peek().split("=", 1)
        values = [value]
    else:
        flag = arg_list.peek()
        values = []

    parsed_options = []

    # Flags can be either long or short (e.g. -l, --long)

    if re.match(r"^--", flag): # Handle long flags
        param = get_param_for_flag(flag)
        arg_list.consume()

        values += _consume_optional_values(arg_list, param.nargs, num_already_consumed_values=len(values))

        parsed_options.append(_ParseResult(param, flag, values))

    elif re.match(r"^-[^-]", flag): # Handle short flags
        # Shorthand flags may be joined together 
        # (e.g: 'ls -la' is equal to 'ls -l -a')
        
        if len(flag) == 2: # We have a single shorthand flag
            param = get_param_for_flag(flag)
            arg_list.consume()

            values += _consume_optional_values(arg_list, param.nargs, num_already_consumed_values=len(values))

            parsed_options.append(_ParseResult(param, flag, values))

        else: # We have muliple shorthand flags
            # None of the combined flags may consume any values. 
            # TODO: Maybe change in future...

            if len(values) > 0:
                _parsing_error(arg_list, "Unexpected value encountered. Combined shorthand "
                                        "options can not be assigned values.")

            all_flags = [f"-{char}" for char in flag[1:]]

            for flag in all_flags:
                arg = get_param_for_flag(flag)
                parsed_options.append(_ParseResult(arg, flag, []))

            arg_list.consume()

    else:
        _parsing_error(arg_list, "Invalid flag encountered. This should never happen.")

    return parsed_options


def _consume_optional_values(arg_list, nargs, num_already_consumed_values=0):
    # This function consumes a number of values from the arg list determined by
    # 'nargs'. It always looks ahead to make sure that no optional flags are consumed.
    
    consumed_values = []

    if nargs == OPTIONAL:
        if arg_list.has_remaining() and _is_value(arg_list.peek()) and num_already_consumed_values == 0:
            consumed_values.append(arg_list.consume())
    elif nargs in [ZERO_OR_MORE, ONE_OR_MORE]:
        while arg_list.has_remaining() and _is_value(arg_list.peek()):
            consumed_values.append(arg_list.consume())
    else:
        if nargs is None:
            nargs = 1
        
        while arg_list.has_remaining() and _is_value(arg_list.peek()) and len(consumed_values) + num_already_consumed_values < nargs:
            consumed_values.append(arg_list.consume())
        
    return consumed_values


def _assign_values_to_positionals(positional_params, positional_values):
    parsed_positionals = []

    num_assigned_values_per_param = collections.defaultdict(lambda: 0)
    # Begin with assigning the minimum number of values required for each param,
    # from left to right.
    
    def get_num_assigned_values():
        return sum(num_assigned_values_per_param.values())

    curr_param_index = 0
    
    while get_num_assigned_values() < len(positional_values) and curr_param_index < len(positional_params):
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
    curr_param_index = 0
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
    
    assert get_num_assigned_values() == len(positional_values), (f"Internal error, too many positional values "
        f"passed to '_assign_values_to_positionals' {get_num_assigned_values()} != {len(positional_values)}.")

    value_indices = list(itertools.accumulate(num_assigned_values_per_param.values()))
    value_indices.insert(0, 0)

    for i, param in enumerate(positional_params):
        parsed_positionals.append(_ParseResult(
            param, 
            None, 
            positional_values[value_indices[min(i, len(value_indices)-1)]:value_indices[min(i+1, len(value_indices)-1)]]
        ))

    return parsed_positionals
