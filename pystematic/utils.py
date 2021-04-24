import contextlib
import os
import logging
import time
import multiprocessing

import torch

from rich.console import Console
from rich.theme import Theme
from rich.markup import escape

logger = logging.getLogger('pystematic_torch')

class PytorchLogHandler(logging.Handler):
    """Handle logging for both single- and multiprocess contexts."""

    def __init__(self, no_style=False):
        super().__init__()
        theme = Theme({
            'debug':    'magenta',
            'info':     'blue',
            'warning':  'yellow',
            'error':    'red',
            'rank': "green",
            'name': "green"

        }, inherit=False)

        if no_style:
            theme = Theme({}, inherit=False)

        self.console = Console(theme=theme)

    def handle(self, record):
        level_str = escape(f"[{record.levelname}]")
        level = f"[{record.levelname.lower()}]{level_str}[/{record.levelname.lower()}]"
        msg = f"{record.getMessage()}"

        name = f"[name]\[{record.name}][/name]"

        if torch.distributed.is_initialized():
            rank = f"[rank][RANK {torch.distributed.get_rank()}][/rank]"
            self.console.print(f"{level} {rank} {name} {msg}")
        else:
            self.console.print(f"{level} {name} {msg}")


@contextlib.contextmanager
def envvars(env):
    original_environ = os.environ.copy()
    os.environ.update(env)
    yield
    os.environ.clear()
    os.environ.update(original_environ)

class ProcessQueue:

    def __init__(self, num_processes, gpu_ids=[], num_gpus_per_process=None):
        self._mp_context = multiprocessing.get_context('spawn')
        self._num_processes = num_processes
        self._num_gpus_per_process = num_gpus_per_process
        self._live_processes = []

        self._gpu_resource = Resource(gpu_ids)

    def _wait(self):
        sentinels = [proc.sentinel for proc in self._live_processes]
        finished_sentinels = multiprocessing.connection.wait(sentinels)

        completed_procs = []
        
        for proc in self._live_processes:
            if proc.sentinel in finished_sentinels:
                completed_procs.append(proc)

        for proc in completed_procs:
            if self._num_gpus_per_process is not None:
                self._gpu_resource.free(proc.gpus)
            self._live_processes.remove(proc)

    def run_and_wait_for_completion(self, experiment, list_of_params):

        for params in list_of_params:

            while len(self._live_processes) >= self._num_processes:
                self._wait()

            if self._num_gpus_per_process is not None:
                gpus = self._gpu_resource.allocate(self._num_gpus_per_process)

                with envvars({"CUDA_VISIBLE_DEVICES": ",".join([str(id) for id in gpus])}):
                    proc = self._mp_context.Process(
                        target=experiment.run,
                        args=(params, )
                    )

                    proc.gpus = gpus

                    proc.start()
                    self._live_processes.append(proc)
            else:
                proc = self._mp_context.Process(
                    target=experiment.run,
                    args=(params, )
                )

                proc.start()
                self._live_processes.append(proc)

        while len(self._live_processes) > 0:
            self._wait()




class ProcessPool:
    pass


class Task:
    pass

class Resource:
    #TODO: add allocation cap and distinct option

    def __init__(self, resource_ids):
        self._resources = {}

        for id in resource_ids:
            self._resources[id] = 0

    def allocate(self, num_resources):
        if num_resources > len(self._resources):
            raise ValueError(f"There are only '{len(self._resources)}' distinct "
                              "resources available. Tried to allocate '{num_resources}'.")
        
        ids_with_least_count = sorted(self._resources.keys(), key=lambda id: self._resources[id])[:num_resources]

        for id in ids_with_least_count:
            self._resources[id] += 1

        return ids_with_least_count

    def free(self, resource_ids):
        for id in resource_ids:
            if self._resources[id] <= 0:
                logger.warn(f"Tried to free resource with id '{id}' which is unallocated (count is '{self._resources[id]}').")
            else:
                self._resources[id] -= 1
