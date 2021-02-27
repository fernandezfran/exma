exma
====

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
