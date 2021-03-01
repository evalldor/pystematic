*A collection of tools that helps you to systematically setup and run reproducible experiments in pytorch.*

This framework has grown out of my own needs as an ML research engineer. The aim
is to assist in the experimentation of and development of all kinds of ML
experiments. More specifically it: 

* Provides a CLI for your experiments 
  
* Helps with managing parameters of you experiments (such as learning rates
  etc.), and makes them assignable from the CLI

* Helps you transition seamlessly between parallel and non-parallel sessions.

* Helps with recording stats and directing your output data to separate folders for different runs


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
