API documentation
=================

Decorators
----------

.. autodecorator:: pystematic.experiment(name=None, inherit_params=None, defaults={}, group=None)

.. autodecorator:: pystematic.parameter

.. autofunction:: pystematic.group(name=None)

Core
----

These classes are not used manually, but created by their corresponding decorators.

.. autoclass:: pystematic.core.Experiment
    :members: add_parameter, get_parameters, run, cli, run_in_new_process
    :undoc-members:

.. autoclass:: pystematic.core.ExperimentGroup

Experiment API
--------------

The experiment API is available for the currently running experiment. The use of
the API when no experiment is running results in undefined behavior. 

Global attributes
+++++++++++++++++

These attributes holds information related to the current experiment. Note that
they are uninitialized until an experiment has actually started.

.. autodata:: pystematic.output_dir
    :annotation: : pathlib.Path

    Holds a :code:`pathlib.Path` object that points to the current output
    directory. All output from an experiment should be written to this folder.
    All internal procedures that produce output will always write it to this
    folder. When you want to output something persistent from the experiment
    yourself, it is your responsibly to use this output directory.


.. autodata:: pystematic.params 
    :annotation: : dict

    Holds a dict of all parameters of the current experiment.



Functions
+++++++++

.. autofunction:: pystematic.new_seed

.. autofunction:: pystematic.launch_subprocess

.. autofunction:: pystematic.run_parameter_sweep

.. autofunction:: pystematic.is_subprocess
