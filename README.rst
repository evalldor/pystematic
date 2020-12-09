*A collection of tools that helps you to systematically setup and run reproducible experiments in pytorch.*

CLI
Build on Click
Define experiment.
Define args.
global_entrypoint
Run script


Context
Holds all state related to experiment
Configuration/params
Default params
Examples in script



Reproducibility
random seed


Training
Looper (optional)


Logging
Backends
adding and commiting.
gobal_step required (= x-axis in all data charts)


Distributed training
Transparent switching between distributed and normal training.
Launching
default args
context helper methods
idendical random seeds

pitfalls with random seeds. (Code conditional on process rank that calls new_seed())



All experiments are registered with pystematic.global_entrypoint

```

```