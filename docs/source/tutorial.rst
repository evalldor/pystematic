Tutorial
========

The central concept in pystematic is that of an *Experiment*. An experiment
consists of:

#. A **main function** that executes the code associated with the experiment,

#. and a set of **parameters**, that controls some aspects of the experiments
   behavior.


Defining experiments
--------------------

Defining an experiment is super easy. You simple decorate the main function of
your experiment with the :code:`experiment` decorator:

.. code-block::

   import pystematic

   @pystematic.experiment
   def my_experiment(params, context):
      print("Hello from my_experiment")

We will discuss the arguments to the function shortly. To be able to run the
experiment from the commandline, you add the following at the bottom of your
source file:

.. code-block::

   if __name__ == "__main__":
      pystematic.global_entrypoint()


Your file now has the capabilities of a full-fledged CLI. The call to
:code:`pystematic.global_entrypoint()` works as a CLI entrypoint to all
experiments that you have defined. When you run the file without arguments you
are presented with a list of all experiments available:

.. code-block:: bash

   $ python path/to/file.py



To run a specific experiment, simply append its name to the commandline:

.. code-block:: bash

   $ python path/to/file.py my-experiment
   Hello from my_experiment


Adding parameters
-----------------

Each experiment has a set of parameters associated with it. If you run:

.. code-block:: bash

   $ python path/to/file.py my-experiment -h

you will see that the experiment we defined is already equipped with a set of
default parameters. To add additional parameters to the experiment, you use the
:code:`parameters` decorator:

.. code-block::

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


A note on naming conventions
----------------------------

At this point it is probably a good idea to mention something about the
naming conventions used. 

You may have noticed that in the python source code, the name of all
experiments and parameters use the snake_case convention, but on the
commandline, these are magically converted to kebab-case. This seems to be a
convention in CLI tools, and this framework sticks to that convention.

To reiterate, this means that on the commanline, all paramters and
experiments use the kebab-case naming convention, but in the source code,
they all use the snake_case naming convention.


Experiment output
-----------------

If you tried out the examples above you might have noticed that a folder named
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
contains the value of all parameters when the experiment was run. This means
that, as long as you write all experiment output to the directory pointed to by
the property :code:`pystematic.output_dir`, you can keep track of which
paramaters gave which output. Neat!