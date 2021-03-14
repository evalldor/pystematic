API documentation
======================================

Decorators and entrypoint
-------------------------

.. autodecorator:: pystematic.experiment

.. autodecorator:: pystematic.parameter

.. autofunction:: pystematic.global_entrypoint

Experiment API
--------------

General
+++++++

.. autodata:: pystematic.output_dir
    :no-value:

.. autodata:: pystematic.params
    :no-value:

.. autodata:: pystematic.context
    :no-value:

.. autofunction:: pystematic.new_seed

.. autofunction:: pystematic.run_experiment

.. autofunction:: pystematic.launch_subprocess

.. autofunction:: pystematic.is_subprocess


Distributed
+++++++++++

.. autofunction:: pystematic.init_distributed

.. autofunction:: pystematic.is_distributed

.. autofunction:: pystematic.is_master

.. autofunction:: pystematic.get_num_processes

.. autofunction:: pystematic.get_rank

.. autofunction:: pystematic.distributed_barrier



Checkpoints
+++++++++++

.. autofunction:: pystematic.save_checkpoint

.. autofunction:: pystematic.load_checkpoint


Components
----------
.. autoclass:: pystematic.Recorder

.. autoclass:: pystematic.Counter

.. autoclass:: pystematic.BetterDataLoader

