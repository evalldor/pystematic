import os
import sys
from rich.table import Table
from rich.padding import Padding
from rich.markup import escape
from rich.text import Text
from rich.console import Console
from rich.theme import Theme
from rich.style import Style

from . import parametric
from . import core

class GroupHelpFormatter(parametric.HelpFormatter):

    def __init__(self, group) -> None:
        super().__init__()

        self._group = group

    def print_help(self, param_manager):
        super().print_help(param_manager)
        
        self._print_experiments()
        self.console.print()

    def print_error(self, error:Exception):
        self.console.print(f"[error]Error:[/error] {error} Use [flag]-h[/flag] for help.")

    def _print_experiments(self):
        self.console.print("[heading]Registered experiments:[/heading]")
        grid = Table(
            box=None,
            padding=(0, 1, 0, 0),
            show_header=False,
            show_footer=False,
            show_edge=False,
            width=80
        )

        grid.add_column()

        self._print_experiments_recursive(grid, self._group.experiments, 0)

        self.console.print(Padding(grid, (0, 1)))

    def _print_experiments_recursive(self, grid, experiments, padding):
        
        for exp in experiments:
            grid.add_row(Padding(exp.name, (0, 2*padding)))

            if isinstance(exp, core.ExperimentGroup):
                self._print_experiments_recursive(grid, exp.experiments, padding+1)


class ExperimentHelpFormatter(parametric.HelpFormatter):

    def print_usage(self, param_manager: parametric.ParameterManager):
        positionals = param_manager.get_cli_positionals()
        positionals_usage = []

        for param in positionals:
            positionals_usage.append(self._format_metavar(param, param.name))
    
        self.console.print("[heading]Usage:[/heading]")
        self.console.print(Padding(f"{os.path.basename(sys.argv[0])} [PARAMETERS] {' '.join(positionals_usage)}", (0, 1)))

    def print_help(self, param_manager: parametric.ParameterManager):
        self.print_usage(param_manager)
        self.console.print()

        self._print_parameters(param_manager)

    def _print_parameters(self, param_manager):

        groups = param_manager.get_groups()
        ungrouped_parameters = param_manager.get_ungrouped_cli_optionals()
        grid = Table(
            box=None,
            padding=(0, 0, 0, 0),
            show_header=False,
            show_footer=False,
            show_edge=False,
            width=80,
        )

        grid.add_column()
        grid.add_column()
        grid.add_column()

        
        self._print_params(grid, ungrouped_parameters)

        for group in groups:
            self._print_group(grid, group)

        self.console.print("[heading]Parameters:[/heading]")
        self.console.print(Padding(grid, (0, 0, 0, 1)))
        self.console.print()

    def _print_group(self, grid, group):
        grid.add_row()
        grid.add_row(f"[heading]{group.name}[/heading]")
        
        self._print_params(grid, group.get_cli_optionals(), left_padding=1)

    def _print_params(self, grid, params, left_padding=0):
        

        for param in params:
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

                grid.add_row(Padding(flags, (0, 0, 0, left_padding)), Padding(nargs, (0, 2)), help)
                grid.add_row()
        

    def print_error(self, error:Exception):
        self.console.print(f"[error]Error:[/error] {error} Use [flag]-h[/flag] for help.")
