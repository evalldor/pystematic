*A collection of tools that helps you to systematically setup and run
reproducible experiments in pytorch.*

TODO
====

- !!python/tuple

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
