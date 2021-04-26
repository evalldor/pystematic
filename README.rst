Pystematic is a framework that helps you to systematically setup and run
reproducible experiments in python. The main concept revolves around defining
experiments together with a set of parameters.

For example, the following code defines an experiment named ``example`` with two
parameters, ``string_param`` and ``int_param``:

.. code-block:: 

    import pystematic.classic as ps
    
    @ps.parameter(
        name="string_param",
        type=str,
        help="A string parameter"
    )
    @ps.parameter(
        name="int_param",
        type=int,
        help="An int parameter",
        default=0
    )
    @ps.experiment
    def example_experiment(params):
        print("Hello from example_experiment.")
        print("string_param is {params['string_param']} and int_param is {params['int_param']}.")


You can run the experiment by either supplying a dict containing the values for
the parameters:

.. code-block:: 

    if __name__ == "__main__":
        example_experiment.run({
            "string_param": "hello",
            "int_param": 10
        })

.. code-block:: 

    $ python path/to/file.py


Or you can run the experiment from the command line:

.. code-block:: 

    if __name__ == "__main__":
        example_experiment.cli()

.. code-block:: 

    $ python path/to/file.py --string-param hello --int-param 10

Full documentation is available at ???.

Note that this project is still in the early stages. There may be some rough
edges.


.. TODO
.. ====
.. - Parameter groups

.. CLI
.. ---
.. - Define experiment.
.. - Define params.
.. - running experiments



.. Reproducibility
.. ---------------
.. - "One seed to rule them all"
.. - random seed, seeding your random number generators
.. - pitfalls with random seeds. (Code conditional on process rank that calls new_seed())


.. Recording
.. ---------
.. - wrapped because of transparency for distributed training
.. - Backends

.. Counters
.. --------

.. Distributed training
.. --------------------
.. - Transparent switching between distributed and normal training.
.. - Launching
.. - default args
.. - context helper methods
.. - identical random seeds

