.. pystematic documentation master file, created by
   sphinx-quickstart on Sat Mar 13 22:32:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pystematic's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:




This framework has grown out of my own needs as an ML research engineer. The aim
is to assist in the experimentation of and development of all kinds of ML
experiments. More specifically it: 

* Makes your experiments easy to run by providing a CLI. 

* Helps with managing parameters of you experiments (such as learning rates
  etc.), and makes them assignable from the CLI

* Helps you transition seamlessly between distributed and non-distributed sessions.

* Helps with recording stats and keeping track of which parameters gave which result.


Experiment
-----------
The central concept in pystematic is that of an *Experiment*. An experiment consists of:

#. A *main function* that executes the code associated with the experiment,
#. and a set of *parameters*, that controls some aspects of the experiments behavior.

Defining an experiment is super easy:

.. code-block::

   import pystematic

   @pystematic.experiment
   def my_experiment(parameters, context):
      print("Hello from my_experiment")

We will discuss the arguments to the function shortly. To be able to run the
experiment from the commandline, you add the following at the bottom of your
source file:

.. code-block::

   if __name__ == "__main__":
      pystematic.global_entrypoint()

The call to `pystematic.global_entrypoint()` works as a CLI entrypoin to all
experiments that you have defined. When you run the file without arguments:

.. code-block:: bash

   $ python path/to/file.py

you are presented with a list of all experiments available:

.. code-block:: bash

   

To run a specific experiment, simply append its name to the commandline:

.. code-block:: bash

   $ python path/to/file.py my-experiment
   