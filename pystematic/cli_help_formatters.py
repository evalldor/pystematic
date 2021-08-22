from rich.table import Table
from rich.padding import Padding
from rich.markup import escape
from rich.text import Text
from rich.console import Console
from rich.theme import Theme

from . import parametric
from . import core

class GroupHelpFormatter(parametric.CliHelpFormatter):

    def __init__(self, group) -> None:
        super().__init__()

        self._group = group

    def print_help(self, positionals, optionals):
        super().print_help(positionals, optionals)
        print()
        self._print_experiments()

    def _print_experiments(self):
        self.console.print("[heading]Registered experiments:[/heading]")
        grid = Table(
            box=None,
            padding=(0, 1, 0, 0),
            show_header=False,
            show_footer=False,
            show_edge=False,
            width=100
        )

        grid.add_column()

        self._print_experiments_recursive(grid, self._group.experiments, 0)

        self.console.print(Padding(grid, (0, 1)))

    def _print_experiments_recursive(self, grid, experiments, padding):
        
        for exp in experiments:
            grid.add_row(Padding(exp.name, (0, 2*padding)))

            if isinstance(exp, core.ExperimentGroup):
                self._print_experiments_recursive(grid, exp.experiments, padding+1)

