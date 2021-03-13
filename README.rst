*A collection of tools that helps you to systematically setup and run
reproducible experiments in pytorch.*

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

* A *main function* that executes the code associated with the experiment,
* and a set of *parameters*, that controls some aspects of the experiments behavior.

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

When you run the file:

.. code-block:: bash
    $ python path/to/file.py

you will find that the call to `pystematic.global_entrypoint()` works as a CLI interface to all experiments
that you have defined.


CLI
---
- Builds on Click
- Define experiment.
- Define args.
- global_entrypoint
- running experiments


Context
-------
- Holds all state related to experiment
- Configuration/params
- Default params
- Examples in script



Reproducibility
---------------
- "One seed to rule them all"
- random seed, seeding you random number generators
- pitfalls with random seeds. (Code conditional on process rank that calls new_seed())



Training
--------
- Looper (optional)


Recording
---------
- wrapped because of transparency for distributed training
- Backends
- adding and commiting.
- gobal_step required (= x-axis in all data charts)

Counters
--------

Distributed training
--------------------
- Transparent switching between distributed and normal training.
- Launching
- default args
- context helper methods
- idendical random seeds




All experiments are registered with pystematic.global_entrypoint
