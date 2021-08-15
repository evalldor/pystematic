Pystematic is a lightweight framework that helps you to systematically setup and
run reproducible computational experiments in python. 

Main features:

* Standardizes how experiments and associated parameters are defined.
  
* Provides both CLI and programmatic interfaces for running your experiments.
  
* Encourages reproducibility by isolating experiment outputs and providing
  tools for managing random seeds.

Quickstart
----------

Installation
============

pystematic is available on pypi, and can be installed with your package manager of choice.

If using pypoetry:

.. code-block:: 

    $ poetry add pystematic

    
Or just pip:

.. code-block:: 

    $ pip install pystematic


Defining and running experiments
================================

Experiments and parameters are defined with decorators. The following example
defines an experiment named ``example_experiment`` with two parameters,
``string_param`` and ``int_param``:

.. code-block:: python

    import pystematic
    
    @pystematic.parameter(
        name="string_param",
        type=str,
        help="A string parameter"
    )
    @pystematic.parameter(
        name="int_param",
        type=int,
        help="An int parameter",
        default=0
    )
    @pystematic.experiment
    def example_experiment(params):
        print("Hello from example_experiment.")
        print(f"string_param is {params['string_param']} and int_param is {params['int_param']}.")


You can run the experiment either by supplying a dict containing the values for
the parameters:

.. code-block:: python

    example_experiment.run({
        "string_param": "hello",
        "int_param": 10
    })

Or you can run the experiment from the command line:

.. code-block:: python

    if __name__ == "__main__":
        example_experiment.cli()


and then from the terminal:

.. code-block:: 

    $ python path/to/file.py --string-param hello --int-param 10

Documentation
-------------

Full documentation is available at ???.

Note that this project is still in the early stages. You may encounter rough
edges.
