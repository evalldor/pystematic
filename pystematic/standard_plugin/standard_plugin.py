import wrapt
import pathlib
import random
import logging
import multiprocessing
import datetime
import string
import functools

from rich.console import Console
from rich.theme import Theme
from rich.markup import escape

from . import yaml_wrapper as yaml
import pystematic.core as core

from .. import parametric


logger = logging.getLogger('pystematic.standard')

class StandardPlugin:

    def __init__(self, app) -> None:
        self.api_object = StandardApi()

        app.on_experiment_created(self.experiment_created)
        app.on_before_experiment(self.api_object._before_experiment)
        app.on_after_experiment(self.api_object._after_experiment)

        self.extend_api(app.get_api_object())


    def experiment_created(self, experiment):
        for param in standard_params:
            experiment.add_parameter(param)
        
        return experiment

    def extend_api(self, api_object):
        for name in dir(self.api_object):
            if not name.startswith("_"):
                setattr(api_object, name, getattr(self.api_object, name))


def _create_log_dir_name(output_dir, experiment_name):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    chars = string.digits + string.ascii_lowercase # + string.ascii_uppercase
    suffix = "".join(random.SystemRandom().choice(chars) for _ in range(6))
    directory = pathlib.Path(output_dir).resolve().joinpath(experiment_name).joinpath(f"{current_time}-{suffix}")

    return directory

class StandardLogHandler(logging.Handler):

    def __init__(self, no_style=False):
        super().__init__()
        theme = Theme({
            'debug':    'magenta',
            'info':     'blue',
            'warning':  'yellow',
            'error':    'red',
            'name':     "green"

        }, inherit=False)

        if no_style:
            theme = Theme({}, inherit=False)

        self.console = Console(theme=theme)

    def handle(self, record):
        level_str = escape(f"[{record.levelname}]")
        level = f"[{record.levelname.lower()}]{level_str}[/{record.levelname.lower()}]"
        msg = f"{record.getMessage()}"

        name = f"[name]\[{record.name}][/name]"

        self.console.print(f"{level} {name} {msg}")

class StandardApi:

    def __init__(self) -> None:
        self.current_experiment = wrapt.ObjectProxy(None)
        self.params: dict = wrapt.ObjectProxy(None)
        self.output_dir: pathlib.Path = wrapt.ObjectProxy(None)
        self.params_file: pathlib.Path = wrapt.ObjectProxy(None)
        self.random_gen: random.Random = wrapt.ObjectProxy(None)

    def _before_experiment(self, experiment, params):
        self.current_experiment.__wrapped__ = experiment
        self.params.__wrapped__ = params
    
        if self.params["debug"]:
            log_level = "DEBUG"
        else:
            log_level = "INFO"

        logging.basicConfig(level=log_level, handlers=[StandardLogHandler()])

        if params["subprocess"]:
            logger.debug(f"Initializing subprocess...")
            self.output_dir.__wrapped__ = pathlib.Path(params["subprocess"]).parent
            self.params_file.__wrapped__ = pathlib.Path(params["subprocess"])
        else:
            self.output_dir.__wrapped__ = _create_log_dir_name(params["output_dir"], experiment.name)
            self.params_file.__wrapped__ = self.output_dir.joinpath("parameters.yml")
        
            if not experiment.no_output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)

                logger.debug(f"Writing parameters file to '{self.params_file}'.")
                with self.params_file.open("w") as f:
                    yaml.dump(params, f)

        self.random_gen.__wrapped__ = random.Random(params["random_seed"])

    def _after_experiment(self):
        procs = multiprocessing.active_children()
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass

        for proc in procs:
            try:
                proc.join()
            except Exception:
                pass

    def new_seed(self, nbits=32) -> int:
        """Use this function to generate random numbers seeded by the experiment
        parameter ``random_seed``. Expected use is to seed your own random number
        generators.
        """
        return self.random_gen.getrandbits(nbits)

    def seed_known_random_generators(self) -> None:
        """This is just a helper to seed all known random modules with
        reproducible seeds."""

        logger.info(f"Seeding python random module")

        random.seed(self.new_seed())
    
    def launch_subprocess(self, **additional_params) -> multiprocessing.Process:
        """Launches a subprocess. The subprocess will have the same output
        directory and parameters as the current process.

        Args:
            **additional_params: Any additional parameters that should be 
                passed to the subprocess. Params given here takes precedence 
                over the parameters copied from the current experiment.

        .. warning:: 

            The subprocess will be initialized with the same random
            seed as the current process. If this is not what you want, you
            should pass a new seed to this function in the ``random_seed`` parameter. 

            E.g.:
            >>> pystematic.launch_subprocess(random_seed=pystematic.new_seed())

        """
        subprocess_params = {name: value for name, value in self.params.items()}

        for name, value in additional_params.items():
            subprocess_params[name] = value

        subprocess_params["subprocess"] = str(self.params_file)

        logger.debug(f"Launching subprocess with arguments '{' '.join(subprocess_params)}'.")

        return self.current_experiment.run_in_new_process(subprocess_params)

    def run_parameter_sweep(self, experiment, list_of_params, max_num_processes=1) -> None:
        """Runs an experiment with a set of different params. At most
        :obj:`max_num_processes` concurrent processes will be used.
        """
        pool = ProcessQueue(max_num_processes)
        pool.run_and_wait_for_completion(experiment, list_of_params)

    def is_subprocess(self) -> bool:
        """Returns true if this process is a subprocess. I.e. it has been
        launched by a call to :func:`launch_subprocess` in a parent process.
        """

        return self.params["subprocess"] is not None


class ParamsFileBehaviour(parametric.DefaultParameterBehaviour):
    
    def on_value(self, param, value: pathlib.Path, result_dict: dict):
        super().on_value(param, value, result_dict)

        if value is not None:
            if not value.exists():
                raise ValueError(f"File does not exist: '{value}'.")

            blacklisted_config_ops = []

            for param in result_dict.get_params():
                if hasattr(param, "allow_from_file") and not param.allow_from_file:
                    blacklisted_config_ops.append(param.name)
            
            with value.open("r") as f:
                params_from_file = yaml.load(f)
            
            for key, value in params_from_file.items():
                if key not in blacklisted_config_ops:
                    result_dict.set_value_by_name(key, value)


standard_params = [
    core.Parameter(
        name="output_dir",
        default="./output",
        help="Parent directory to store all run-logs in. Will be created if it does not exist.",
        type=str
    ),
    core.Parameter(
        name="debug",
        default=False,
        help="Sets debug flag on/off.",
        type=bool,
        is_flag=True
    ),
    core.Parameter(
        name="params_file",
        type=pathlib.Path,
        help="Read experiment parameters from FILE.",
        behaviour=ParamsFileBehaviour(),
        allow_from_file=False
    ),
    core.Parameter(
        name="random_seed",
        default=functools.partial(random.getrandbits, 32),
        help="The value to seed the master random number generator with.",
        type=int, 
        default_help="<randomly generated>"
    ),
    core.Parameter(
        name="subprocess",
        default=None,
        help="Internally used to indicate that this process is a subprocess. DO NOT USE MANUALLY.",
        type=pathlib.Path,
        allow_from_file=False,
        hidden=True
    ),
]


class ProcessQueue:

    def __init__(self, num_processes):
        self._mp_context = multiprocessing.get_context('spawn')
        self._num_processes = num_processes
        self._live_processes = []

    def _wait(self):
        sentinels = [proc.sentinel for proc in self._live_processes]
        finished_sentinels = multiprocessing.connection.wait(sentinels)

        completed_procs = []
        
        for proc in self._live_processes:
            if proc.sentinel in finished_sentinels:
                completed_procs.append(proc)

        for proc in completed_procs:
            self._live_processes.remove(proc)

    def run_and_wait_for_completion(self, experiment, list_of_params):

        for params in list_of_params:

            while len(self._live_processes) >= self._num_processes:
                self._wait()
            
            proc = experiment.run_in_new_process(params)
            self._live_processes.append(proc)
            
        while len(self._live_processes) > 0:
            self._wait()
