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

If using poetry:

.. code-block:: 

    $ poetry add pystematic

    
or just pip:

.. code-block:: 

    $ pip install pystematic


Defining and running experiments
================================

Experiments and parameters are defined with decorators. The following example
defines an experiment named ``hello_world`` with a single parameter ``name``:

.. code-block:: python

    @ps.parameter(
        name="name",
        type=str,
        help="The name to greet.",
        required=True
    )
    @ps.experiment
    def hello_world(params):
        print(f"Hello {params['name']}!")


You can run the experiment either by supplying a dict containing the values for
the parameters:

.. code-block:: python

    hello_world.run({
        "name": "World",
    })

or you can run the experiment from the command line by ivoking the ``cli()``
method of the experiment:

.. code-block:: python

    if __name__ == "__main__":
        hello_world.cli()


Then from the terminal you simply run:

.. code-block:: 

    $ python path/to/file.py --name "World"


Documentation
-------------

Full documentation is available at `<https://pystematic.readthedocs.io>`_.

Extensions
----------

For running machine learning experiments in pytorch check out the
`pystematic-torch <https://github.com/evalldor/pystematic-torch>`_ plugin.





Related tools
-------------

Other related tools that might interest you:

* `Aim <https://github.com/aimhubio/aim>`_: record, search and compare 1000s of
  ML training runs.

* `Hydra <https://github.com/facebookresearch/hydra>`_: a framework for elegantly
  configuring complex applications.

* `MLflow <https://github.com/mlflow/mlflow>`_: a machine learning lifecycle platform.
