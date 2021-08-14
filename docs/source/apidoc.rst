API documentation
=================

Decorators
----------

.. autodecorator:: pystematic.experiment(name=None, inherit_params=None, defaults={}, group=None)

.. autodecorator:: pystematic.parameter

.. autofunction:: pystematic.group(name=None)


Experiment API
--------------


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



General
+++++++

.. autofunction:: pystematic.new_seed

.. autofunction:: pystematic.launch_subprocess

.. autofunction:: pystematic.is_subprocess

.. autofunction:: pystematic.torch.run_parameter_sweep


Distributed
+++++++++++

.. autofunction:: pystematic.torch.init_distributed

.. autofunction:: pystematic.torch.is_distributed

.. autofunction:: pystematic.torch.is_master

.. autofunction:: pystematic.torch.get_num_processes

.. autofunction:: pystematic.torch.get_rank

.. autofunction:: pystematic.torch.distributed_barrier


Checkpoints
+++++++++++

.. autofunction:: pystematic.torch.save_checkpoint

.. autofunction:: pystematic.torch.load_checkpoint


Recording
---------

.. autoclass:: pystematic.torch.Recorder

    .. autoproperty:: count

    .. automethod:: step
    
    .. automethod:: scalar

    .. automethod:: image

    .. automethod:: figure

    .. automethod:: state_dict

    .. automethod:: load_state_dict


Torch context
-------------
    
.. autoclass:: pystematic.torch.ContextObject

    
Components
----------

.. autoclass:: pystematic.torch.BetterDataLoader

