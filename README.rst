.. image:: https://img.shields.io/pypi/pyversions/pystematic?style=for-the-badge
    :target: https://pypi.org/project/pystematic/

.. image:: https://img.shields.io/pypi/v/pystematic?style=for-the-badge
    :target: https://pypi.org/project/pystematic/

.. image:: https://img.shields.io/github/workflow/status/evalldor/pystematic/Test?style=for-the-badge
    :target: https://github.com/evalldor/pystematic/actions/workflows/test.yaml

.. image:: https://readthedocs.org/projects/pystematic/badge/?style=for-the-badge
    :target: https://pystematic.readthedocs.io


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

pystematic is available on `pypi <https://pypi.org/project/pystematic/>`_, and
can be installed with your package manager of choice.

If using poetry:

.. code-block:: 

    $ poetry add pystematic

    
or with pip:

.. code-block:: 

    $ pip install pystematic


Defining and running experiments
================================

Experiments and parameters are defined by decorating the main function of the
experiment. The following example defines an experiment named ``hello_world``
with a single parameter ``name``:

.. code-block:: python

    import pystematic as ps
    
    @ps.parameter(
        name="name",
        help="The name to greet.",
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

or you can run the experiment from the command line by invoking the ``cli()``
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

* `MLflow <https://github.com/mlflow/mlflow>`_: a machine learning lifecycle platform.

* `Hydra <https://github.com/facebookresearch/hydra>`_: a framework for
  elegantly configuring complex applications.

* `Click <https://github.com/pallets/click>`_: Python composable command line interface toolkit.
