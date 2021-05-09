.. pystematic documentation master file, created by
   sphinx-quickstart on Sat Mar 13 22:32:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pystematic's documentation!
======================================

This framework has grown out of my own needs as an ML research engineer. The aim
is to assist in defining and running reproducible numerical experiments in
**pytorch**. More specifically it: 

* Defines a standard way to declare experiments and all associated parameters. 

* Provides a CLI for running experiments from the command line.

* Helps keeping track of which parameters gave which result by assigning a
  unique output directory to every run of an experiment.

.. toctree::
   :maxdepth: 2

   tutorial
   apidoc
