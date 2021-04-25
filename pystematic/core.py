import sys
import functools
import multiprocessing
import importlib

from . import parametric


class Experiment:

    def __init__(self, api_object, main_function, default_params=None, name=None, defaults_override={}):
        self.api_object = api_object
        self.default_params = default_params
        self.main_function = main_function
        self.name = name or main_function.__name__.lower().replace("_", "-")
        
        self.param_manager = parametric.ParameterManager(
            defaults_override=defaults_override,
            add_cli_help_option=True
        )

        if self.default_params is not None:
            for param in self.default_params:
                self.add_parameter(param)

        if hasattr(main_function, "__params_memo__"):
            for param in main_function.__params_memo__:
                self.add_parameter(param)

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
            self.api_object.init_experiment(self, params)
            self.main_function(params)
        finally:
            self.api_object.cleanup()


def _run_experiment_by_name(experiment_module, experiment_name, params):
    # used by Experiment.run_in_new_process
    module = importlib.import_module(experiment_module)
    getattr(module, experiment_name).run(params)


class ExperimentGroup:

    def __init__(self, experiment_decorator, main_function, name=None):
        
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

