API documentation
======================================

Decorators and entrypoint
-------------------------

.. autodecorator:: pystematic.experiment

.. autodecorator:: pystematic.parameter

.. autofunction:: pystematic.global_entrypoint


Experiment API
--------------


Global attributes
+++++++++++++++++

These attributes holds information related to the current experiment. Note that
they are uninitialized until an experiment has actually started.

.. autodata:: pystematic.output_dir
    :annotation: : pathlib.Path

    Holds a :code:`pathlib.Path` object that points to the current output
    directory. All output from an experiment should be written to this
    folder.


.. autodata:: pystematic.params 
    :annotation: : dict

    Holds the parameters of the current experiment


.. autodata:: pystematic.context
    :no-value:

    Holds the context object for the current experiment.

General
+++++++

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
