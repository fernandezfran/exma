.. exma documentation master file, created by
   sphinx-quickstart on Thu Feb 25 16:28:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to exma's documentation!
================================

**exma** is an extendable molecular analyzer. 

This project is mainly developed with the objective of analyzing molecular dynamics trajectories using different tools that characterize both the configurational properties and the dynamics of the system. It is intended to analyze monatomic materials or alloys.


Features
--------

With *exma* you can:

* Read and write MD trajectory frames in .xyz and .lammpstrj format (including charges, image of the particles or some other property).
* Estimate errors with the block average method.
* Initializate positions of atoms in SC, BCC or FCC crystals.
* Apply PBC.
* Calculate:
    - clusterization,
    - the coordination number (CN), the ligancy or in a shell, for monoatomic or diatomic systems,
    - effective neighbors (EN),
    - mean square displacement (MSD),
    - the radial distribution function (RDF) of monoatomic or diatomic systems,
    - short range order (SRO): Warren-Cowley parameter (WCP).


GitHub Repository
-----------------

https://github.com/fernandezfran/exma


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorial
   exma
   license
   contributing
   authors
   history

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
