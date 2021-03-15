.. pystematic documentation master file, created by
   sphinx-quickstart on Sat Mar 13 22:32:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pystematic's documentation!
======================================

This framework has grown out of my own needs as an ML research engineer. The aim
is to assist in defining and running reproducible ML experiments in **pytorch**. More
specifically it: 

* Makes your experiments easy to run by providing a CLI. 

* Helps with managing parameters of you experiments (such as learning rates
  etc.), and makes them assignable from the CLI

* Helps you transition seamlessly between distributed and non-distributed sessions.

* Helps with recording stats and keeping track of which parameters gave which result.

.. toctree::
   :maxdepth: 2

   tutorial
   apidoc
