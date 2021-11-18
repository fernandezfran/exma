exma
====

**exma** is an extendable molecular analyzer. 

This project is mainly developed with the objective of analyzing molecular dynamics trajectories using different tools that characterize both the configurational properties and the dynamics of the system. It is intended to analyze monatomic materials or alloys.


Features
--------

With *exma* you can:

* Read and write MD trajectory frames in .xyz and .lammpstrj format (including charges, image of the particles or some other property).
* Write input file for LAMMPS.
* Estimate errors with the block average method.
* Initializate positions of atoms in:
    - SC, BCC, FCC, DCC crystals,
    - spherical nanoparticles.
    - replicate crystal structures.
* Apply PBC.
* Calculate:
    - clusterization (dbscan),
    - the coordination number (CN), the ligancy or in a shell, for monoatomic or diatomic systems,
    - effective neighbors (EN),
    - mean square displacement (MSD),
    - short range order, an amorphous parameter, defined `here`_,
    - the radial distribution function (RDF) of monoatomic or diatomic systems.

.. _here: https://doi.org/10.1039/D1CP02216D

GitHub Repository
-----------------

https://github.com/fernandezfran/exma
