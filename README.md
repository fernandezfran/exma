# exma

> **exma** is an extendable molecular analyzer.

[![Build Status](https://travis-ci.com/fernandezfran/exma.svg?branch=master)](https://travis-ci.com/github/fernandezfran/exma)
[![Documentation Status](https://readthedocs.org/projects/exma/badge/?version=latest)](https://exma.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/exma)](https://pypi.org/project/exma/)

This project is mainly developed with the objective of analyzing molecular dynamics trajectories using different tools that characterize both the configurational properties and the dynamics of the system. It is intended to analyze monatomic materials or alloys.


## Features

With _exma_ you can:

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
    - short range order, an amorphous parameter, defined [here](https://doi.org/10.1039/D1CP02216D),
    - the radial distribution function (RDF) of monoatomic or diatomic systems.


## Requirements

_exma_ requires Python 3.8 to run, the others specific requirements are provided during installation. gcc 9.3 is necesary to compile the shared libreries if you are going to run `tox`.


## Installation

### Stable release

To install the most recent stable release of _exma_ with [pip](https://pip.pypa.io/en/stable/), run the following command in your termninal:

```bash
pip3 install exma
```

### From sources

To installing _exma_ from sources you can clone this [GitHub repo](https://github.com/fernandezfran/exma) 

```bash
git clone https://github.com/fernandezfran/exma.git
```

and inside your local directory install it in the following way 

```bash
pip3 install .
```

or

```bash
python3 setup.py install
```

## Usage

To use the extendable molecular analyzer in a project:

```python
import exma
```

There are different examples in the documentation [tutorials](https://exma.readthedocs.io/en/latest/tutorial.html).

## Release History

[HISTORY](https://github.com/fernandezfran/exma/blob/master/HISTORY.md)


## License

[MIT License](https://github.com/fernandezfran/exma/blob/master/LICENSE)


## TODO

[TODO List](https://github.com/fernandezfran/exma/blob/master/TODO.md)


## Contributing

[Contributing](https://github.com/fernandezfran/exma/blob/master/CONTRIBUTING.md)
