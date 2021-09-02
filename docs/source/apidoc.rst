API reference
=============

Decorators
----------

.. autodecorator:: pystematic.experiment

.. autodecorator:: pystematic.group

.. autodecorator:: pystematic.parameter

.. autodecorator:: pystematic.param_group(name, help=None, *parameters)


Core types
----------

These classes are not supposed to be instantiated manually, but only through
their corresponding decorators.

.. autoclass:: pystematic.core.Experiment
  :members: add_parameter, get_parameters, run, cli, run_in_new_process
  :undoc-members:

.. autoclass:: pystematic.core.ExperimentGroup
  :members: experiment, group, cli, add_parameter, get_parameters
  :undoc-members:

.. autoclass:: pystematic.core.Parameter
  :members: 
  :undoc-members:

.. autoclass:: pystematic.core.ParameterGroup
  :members: 
  :undoc-members:

.. autoclass:: pystematic.core.PystematicApp
  :members: get_api_object, on_experiment_created, on_before_experiment, on_after_experiment
  :undoc-members:


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

  Holds a dict of all parameters of the current experiment. It is the same
  dict that is passed to the main function. **You should never modify this
  dict.**


Functions
+++++++++

.. autofunction:: pystematic.new_seed

.. autofunction:: pystematic.launch_subprocess

.. autofunction:: pystematic.run_parameter_sweep

.. autofunction:: pystematic.is_subprocess

.. autofunction:: pystematic.local_rank

.. autofunction:: pystematic.param_matrix


Default parameters
------------------

The following parameters are added to all experiments by default. Note that
these are also listed if you run an experiment from the command line with the
``--help`` option.

* ``output_dir``: Parent directory to store all run-logs in. Will be created if
  it does not exist. Default value is ``./output``. 

* ``random_seed``: The value to seed the master random number generator with.
  Default is randomly generated.

* ``params_file``: Read experiment parameters from a yaml file, such as the one
  dumped in the output dir from an eariler run. When this option is set from the
  command line, any other options supplied after will override the ones
  loaded from the file.

* ``debug``: Sets debug flag ON/OFF. Configures the python logging mechanism to
  print all DEBUG messages. Default value is ``False``.
