*A collection of tools that helps you to systematically setup and run
reproducible experiments in pytorch.*

TODO
====

- Parameter groups
- make context manually created. 
- Make checkpoint loading manual. Add global property checkpoint
- Make proper experiment decorator
- Make clearer separation between pytorch specifics and general stuff

CLI
---
- Define experiment.
- Define params.
- running experiments



Reproducibility
---------------
- "One seed to rule them all"
- random seed, seeding your random number generators
- pitfalls with random seeds. (Code conditional on process rank that calls new_seed())


Recording
---------
- wrapped because of transparency for distributed training
- Backends

Counters
--------

Distributed training
--------------------
- Transparent switching between distributed and normal training.
- Launching
- default args
- context helper methods
- identical random seeds

