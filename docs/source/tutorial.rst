Tutorial
========

Defining experiments
--------------------

The central concept in pystematic is that of an *Experiment*. An experiment
consists of:

#. A **main function** that executes the code associated with the experiment,

#. a set of **parameters**, that controls aspects of the experiments
   behavior.

To define an experiment you simply decorate the main function of
your experiment with the :func:`pystematic.experiment` decorator:

.. code-block:: python

   import pystematic

   @pystematic.experiment
   def my_experiment(params):
      print("Hello from my_experiment")

The main function must take a single argument which is a dict containing the
experiment parameters. 



To run the experiment you have a couple of different options. The simplest one
is to run the experiment by with the :meth:`pystematic.core.Experiment.run` method:

.. code-block:: python

   my_experiment.run({})

it takes a single argument which is a dict of parameter values, as we haven't
defined any parameters yet, we'll pass an empty dict for now.


Another option is to run the experiment from the command line. To do that, we
call the :meth:`pystematic.core.Experiment.cli` method:

.. code-block:: python

   if __name__ == "__main__":
      my_experiment.cli()

The file now has the capabilities of a full-fledged CLI. When you run the file
from the command line:

.. code-block:: bash

   $ python path/to/file.py 
   Hello from my_experiment

you will see that the experiment is run.

.. note::
   At most one experiment can be active at a time. This mean that if you want to
   run an experiment from within another experiment, you need to start the new
   experiment in a new process, which can be done with the method
   :meth:`pystematic.core.Experiment.run_in_new_process`.

Adding parameters
-----------------

Every experiment has a set of parameters associated with it. If you run:

.. code-block:: bash

   $ python path/to/file.py -h

you will see that the experiment we defined earlier is already equipped with a
set of :ref:`default parameters <apidoc:default parameters>`. To add additional
parameters to the experiment, you use the :func:`pystematic.parameter`
decorator:

.. code-block:: python

   import pystematic

   @pystematic.parameter(
      name="string_to_print",
      type=str,
      help="This string will be printed when the experiment is run",
      default="No string was given",
   )
   @pystematic.experiment
   def my_experiment(params):
      print(f"string_to_print is {params['string_to_print']}")

The code above adds a string parameter named ``string_to_print`` with a default
value, and a description of the parameter. When we run the experiment - either
programmatically or from the command line - we can set a value for the
parameter.

A note on naming conventions
----------------------------

At this point it is probably a good idea to mention something about the
naming conventions used. 

You may have noticed that in the python source code, the name of all experiments
and parameters use the snake_case convention, but on the command line, these are
magically converted to kebab-case. This seems to be a convention in CLI tools,
and this framework sticks to that convention.

To reiterate, this means that on the command line, all paramters and
experiments use the kebab-case naming convention, but in the source code,
they all use the snake_case naming convention.


Experiment output
-----------------

If you tried running the examples above you might have noticed that a folder named
``output`` was created in you current working directory. This is no accident.
Every time an experiment is run, a unique output folder is created in the
configured output directory. The folder creation follows the naming convention
``<output_dir>/<experiment_name>/<current date and time>``, where ``output_dir``
is the value of the parameter with the same name (which defaults to your current
working directory).

The reason each invocation of an experiment gets its own output directory is to
avoid mixing up outputs from different runs.

If you look into the output directory of one of the experiment runs you will
also notice that there is a file there named ``parameters.yaml``. This file
contains the value of all parameters when the experiment was run. 

When an experiment is run, this newly created output directory is bound to the
:data:`pystematic.output_dir` property. All data that you want to output from
the experiment should be written to this directory. It is your responsibly to
make sure that relevant output is written to this directory.


Managing random numbers
-----------------------

Reproducibility is an integral part of any sort of research. One of the default
parameters added to all experiments is an integer named ``random_seed``. If a
value for this parameter is not supplied when an experiment is run, a random
value will be generated and assigned to this parameter. The value of the
``random_seed`` parameter is used to seed an internal random number generator
used by pystematic. Whenever you need to seed a random number generator in your
experiment, you call the function :func:`pystematic.new_seed` to obtain a seed.

Internally, the :func:`pystematic.new_seed` function uses the internal number
generator to generate a new number every time it is called. This way, you make
the experiment reproducible by controlling all sources of randomness in the
experiment with the single "global" seed provided in the ``random_seed``
parameter. 

Here's how a simple experiment might make sure that random numbers are
reproducible:

.. code-block:: python

   import random

   import numpy as np
   import pystematic as ps


   @ps.experiment
   def reproducible_experiment(params):
      random.seed(ps.new_seed())
      np.random.seed(ps.new_seed())
      # etc.


Grouping experiments
--------------------

If you have several experiments defined in the same file, you may want to be
able to run them all from the CLI without changing your code. This is what
groups are for.

Take the following as an example:

.. code-block:: python

   import pystematic as ps


   @ps.experiment
   def prepare_dataset(params):
      # ...

   @ps.experiment
   def fit_model(params):
      # ...
   
   @ps.experiment
   def visualize_results(params):
      # ...

   if __name__ == "__main__":
      # prepare_dataset.cli()
      # fit_model.cli()
      visualize_results.cli()


The code above has three defined experiments. We can run them from the cli by
calling each experiments ``cli()`` function, but that would require us to change
the code whenever we want to run another experiment. To remedy this, we can add
them all to a group. We first use the :func:`pystematic.group` decorator to
define the group, and then use to group's own experiment decorator to define the
experiments, instead of the global experiment decorator. We then use the group's
``cli()`` function to activate the cli:

.. code-block:: python

   import pystematic as ps

   @ps.group
   def my_group():
      pass

   @my_group.experiment # <--- Note that we are using the groups experiment 
                        #      decorator instead of the global one.
   def prepare_dataset(params):
      # ...

   @my_group.experiment
   def fit_model(params):
      # ...
   
   @my_group.experiment
   def visualize_results(params):
      # ...

   if __name__ == "__main__":
      my_group.cli()


We can now choose which experiment to run like this:

.. code-block:: bash

   $ python path/to/file.py prepare-dataset <experiment params here>
   # or:
   $ python path/to/file.py fit-model <experiment params here>
   # or:
   $ python path/to/file.py visualize-results <experiment params here>


To get a list of all available experiments simply run the script with the ``-h``
flag:

.. code-block:: bash

   $ python path/to/file.py -h


Another feature of groups is that all parameters that you add to the group will
be inherited by all experiments in the group. This is very useful when you have
several experiments take share a set of common parameters. By adding the common
parameters to the group, you don't have to repeat the parameter definitions for
every experiment. In the example above, we could define a common parameters like
this:

.. code-block:: python

   import pystematic as ps

   @pystematic.parameter(
      name="dataset_path",
      type=str
   )
   @ps.group
   def my_group():
      pass

Groups can be arbitrarily nested to create hierarchies of experiments. Note that
the main function of the group is never run. It is only used as a symbolic
convenience for defining the group.


Extensions
----------

Pystematic is built from the core to be extensible. See the page on
:ref:`extending:writing extensions` to learn how you can design and customize
experiments of your own.


To be continued